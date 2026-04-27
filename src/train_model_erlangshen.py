import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments, \
    AutoTokenizer
from datasets import Dataset

# 设置路径 bert-base-multilingual-cased
model_path = "/hy-tmp/zyh/model/Erlangshen-DeBERTa-v2-320M-Chinese"
data_path = "/hy-tmp/zyh/data/讨论实验数据-5000.csv"
save_path = "/hy-tmp/zyh/save/5000_zero"

# 创建保存目录
os.makedirs(save_path, exist_ok=True)

# 读取数据
df = pd.read_csv(data_path)
# df = df.drop('label', axis=1)
df = df.rename(columns={'result_label': 'label'})

positive_samples = df[df['label'] == 1]  # 1表示违规，0表示不违规
negative_samples = df[df['label'] == 0]

# 采样使正负样本比例为25%:75%
# 计算需要的负样本数量
negative_size = int(len(positive_samples) * 3)  # 75% / 25% = 3
print(f"negative_size: {negative_size}")
if negative_size > len(negative_samples):
    # 如果负样本不足，使用全部负样本
    negative_samples = negative_samples.sample(frac=1, random_state=42)
else:
    # 否则采样所需数量的负样本
    negative_samples = negative_samples.sample(n=negative_size, random_state=42)

# 合并样本
balanced_df = pd.concat([positive_samples, negative_samples])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 划分训练集和验证集
train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42)


# ---------------------- 添加数据分布查看代码 ----------------------
def print_data_distribution(df, dataset_name):
    """打印数据集的标签分布情况"""
    total = len(df)
    positive_count = df[df['label'] == 1].shape[0]
    negative_count = total - positive_count
    positive_ratio = positive_count / total * 100
    negative_ratio = 100 - positive_ratio

    print(f"\n{dataset_name} 数据分布:")
    print(f"总样本数: {total}")
    print(f"正样本(1)数量: {positive_count}，占比: {positive_ratio:.2f}%")
    print(f"负样本(0)数量: {negative_count}，占比: {negative_ratio:.2f}%")


# 查看平衡后的数据分布
print_data_distribution(balanced_df, "平衡后总数据集")

# 查看训练集和验证集的数据分布
print_data_distribution(train_df, "训练集")
print_data_distribution(val_df, "验证集")
# -----------------------------------------------------------------

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = DebertaV2ForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    problem_type="single_label_classification"
)

# 准备数据集
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 设置训练参数
training_args = TrainingArguments(
    output_dir=save_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 训练模型
train_result = trainer.train()
trainer.save_model(save_path)

# 评估模型
train_metrics = trainer.evaluate(train_dataset)
val_metrics = trainer.evaluate(val_dataset)

print("\n训练集评估结果:")
for key, value in train_metrics.items():
    print(f"{key}: {value:.4f}")

print("\n验证集评估结果:")
for key, value in val_metrics.items():
    print(f"{key}: {value:.4f}")