import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Union, Any


class PromptGuard:
    def __init__(self, tokenizer_path="../resources/tokenizer/Erlangshen-DeBERTa-v2-320M-Chinese",
                 model_path="../resources/model/new_zero_erlangshen"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # 检查并使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

    def predict_prompt(self, prompt, return_confidence_only=True):
        """
        对单条提示词进行恶意性预测，并返回预测结果或置信度。

        参数:
            prompt (str): 需要预测的提示词字符串。
            return_confidence_only (bool): 是否仅返回恶意性置信度。默认为True。

        返回:
            float 或 tuple:
                - 如果 return_confidence_only 为 True，返回恶意性置信度（float）。
                - 如果 return_confidence_only 为 False，返回一个元组 (预测类别, 恶意性置信度)。

        异常:
            ValueError: 当计算出的恶意性置信度不在 [0, 1] 范围内时抛出。
        """
        # 对单条提示词进行分词
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # 将输入数据移到GPU
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # 模型预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 获取预测结果
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        label2id = self.model.config.label2id
        malicious_id = label2id.get('LABEL_1', label2id.get('malicious', 1))  # 多种可能的标签名
        malicious_confidence = probs[0][malicious_id].item()

        if not (0 <= malicious_confidence <= 1):
            raise ValueError(f"Invalid confidence score: {malicious_confidence}")

        return malicious_confidence if return_confidence_only else (pred_class, malicious_confidence)

    def batch_predict(self, prompts: list, threshold: float = 0.5) -> list:
        """
        批量处理提示词，返回每个提示词的预测结果。

        参数:
            prompts (list): 包含多个提示词的列表，每个提示词为字符串类型。
            threshold (float): 判断恶意内容的置信度阈值，默认为0.5。当模型输出的置信度大于该阈值时，
                              对应提示词被判定为恶意内容（返回1），否则为非恶意内容（返回0）。

        返回:
            list: 预测结果列表，每个元素为0或1，表示对应提示词是否为恶意内容。
        """
        # 将输入提示词转换为模型可接受的张量格式，并进行填充和截断处理
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # 将输入数据移到GPU
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # 使用模型进行推理，禁用梯度计算以提高效率
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取模型输出的logits（若存在），否则直接使用outputs
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # 对logits应用softmax函数，得到各类别的概率分布
        probs = torch.softmax(logits, dim=1)

        # 获取模型配置中的标签映射关系，确定恶意内容对应的类别ID
        label2id = self.model.config.label2id
        malicious_id = label2id.get('LABEL_1', label2id.get('malicious', 1))

        # 提取所有提示词中恶意内容类别的置信度
        confidences = probs[:, malicious_id].tolist()

        # 根据设定的阈值将置信度转换为二分类结果（0或1）
        return [1 if conf > threshold else 0 for conf in confidences]

    def block_prompt(self, prompt: str, threshold: float = 0.5) -> List[str]:
        """
        :param prompt: 输入提示词，只能是单条提示词
        :param threshold: 违规置信度阈值，超过此值的提示词将被拦截，默认0.5
        :return: 违规返回1，否则返回0
        """
        # 处理空输入
        if not prompt:
            return "无提示词"

        # 获取提示词的置信度
        confidence = self.predict_prompt(prompt)

        if confidence > threshold:
            return 1
        else:
            return 0