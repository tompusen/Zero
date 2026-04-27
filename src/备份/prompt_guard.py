import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
class PromptGuard:
    def __init__(self, tokenizer_path="../resources/tokenizer/Erlangshen-DeBERTa-v2-320M-Chinese", model_path="../resources/model/new_zero_erlangshen"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # 设置为评估模式

    def predict_prompts(self, prompts, return_confidence_only=True):
        """
        :param prompts: 输入提示词列表
        :param return_confidence_only: 是否只返回违规置信度
        :return: 若return_confidence_only为True，返回违规置信度；否则返回(预测类别, 违规置信度)元组
        """
        results = []
        for prompt in prompts:
            # 对单个提示词进行分词
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
            # 模型预测
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 获取预测结果
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            # 获取违规类别(1)的置信度
            malicious_confidence = probs[0][1].item()
            if return_confidence_only:
                results.append(malicious_confidence)
            else:
                results.append((pred_class, malicious_confidence))
        return results

    def predict_prompt(self, prompt, return_confidence_only=True):
        """
        预测单条提示词的违规概率
        :param prompt: 输入的单条提示词
        :param return_confidence_only: 是否只返回违规置信度
        :return: 若return_confidence_only为True，返回违规置信度；否则返回(预测类别, 违规置信度)元组
        """
        # 对单条提示词进行分词
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
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

    from typing import List, Union, Any

    def block_prompts(self, prompts: Union[str, List[str]], threshold: float = 0.5) -> List[str]:
        """
        过滤违规提示词，返回正常提示词列表

        :param prompts: 输入提示词，可以是单条字符串或字符串列表
        :param threshold: 违规置信度阈值，超过此值的提示词将被拦截，默认0.5
        :return: 正常提示词列表，即使输入是单条提示词也返回列表

        说明：
        - 对于单条提示词，返回包含该提示词的列表（如果正常）或空列表（如果违规）
        - 对于提示词列表，返回所有正常的提示词组成的列表
        - 内部通过调用predict_prompts或predict_prompt方法获取置信度
        """
        # 处理空输入
        if not prompts:
            return []

        # 统一转为列表处理，简化逻辑
        prompt_list = [prompts] if isinstance(prompts, str) else prompts

        # 验证输入类型
        if not isinstance(prompt_list, list) or not all(isinstance(p, str) for p in prompt_list):
            raise TypeError("输入必须是字符串或字符串列表")

        # 获取所有提示词的置信度
        confidences = self.predict_prompts(prompt_list)

        normal_prompts = []

        # 过滤违规提示词
        for prompt, confidence in zip(prompt_list, confidences):
            if confidence >= threshold:
                print(f"提示词违规，已拦截！\n提示词: {prompt}\n违规置信度: {confidence:.4f}\n")
                return None
            else:
                normal_prompts.append(prompt)
        print("提示词置信度为：", confidences)
        return normal_prompts


