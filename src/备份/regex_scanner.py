import json

import yaml
import re
import glob
import os


class RegexScanner:
    def alone_check_text(self, input_string, rules_file_path="../resources/rules/综合规则库/综合规则.json"):
        """
        检查输入字符串是否匹配综合规则库中的规则

        :param input_string: 待检查的输入字符串
        :param rules_file_path: 综合规则库文件路径
        :return: 如果匹配到规则则返回Description内容，否则返回原字符串
        """
        try:
            # 读取综合规则库
            with open(rules_file_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            # 遍历所有规则
            for rule in rules:
                if 'Regex' in rule and 'Description' in rule:
                    try:
                        # 编译正则表达式并检查匹配
                        pattern = re.compile(rule['Regex'])
                        if pattern.search(input_string):
                            # 如果匹配到规则，返回Description内容
                            return rule['Description']
                    except re.error as e:
                        # 正则表达式编译失败时跳过该规则
                        print(f"正则表达式编译失败: {rule['Regex']}, 错误: {e}")
                        continue

            # 如果没有匹配到任何规则，返回原字符串
            return input_string

        except FileNotFoundError:
            print(f"规则文件未找到: {rules_file_path}")
            return input_string
        except json.JSONDecodeError:
            print(f"规则文件格式错误: {rules_file_path}")
            return input_string
        except Exception as e:
            print(f"检查规则时发生错误: {e}")
            return input_string

    def union_check_text(self, input_string, rules_file_path="../resources/rules/综合规则库/综合规则.json",
                         initial_threshold=0.5, threshold_increment=0.15, threshold_decrement=0.15):
        """
        检查输入字符串是否匹配综合规则库中的规则，并根据匹配情况调整阈值

        :param input_string: 待检查的输入字符串
        :param rules_file_path: 综合规则库文件路径
        :param initial_threshold: 初始阈值，默认为0.5
        :param threshold_increment: 匹配成功-0.15
        :param threshold_decrement: 匹配失败+0.15
        :return: 如果匹配到规则则返回Description内容，否则返回原字符串；同时返回调整后的阈值
        """
        threshold = initial_threshold

        try:
            # 读取综合规则库
            with open(rules_file_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)

            # 遍历所有规则
            matched = False
            for rule in rules:
                if 'Regex' in rule and 'Description' in rule:
                    try:
                        # 编译正则表达式并检查匹配
                        pattern = re.compile(rule['Regex'])
                        if pattern.search(input_string):
                            # 如果匹配到规则，调整阈值并标记匹配状态
                            threshold -= threshold_decrement
                            matched = True

                    except re.error as e:
                        # 正则表达式编译失败时跳过该规则
                        print(f"正则表达式编译失败: {rule['Regex']}, 错误: {e}")
                        continue

            # 如果没有匹配到任何规则，调整阈值
            if not matched:
                threshold += threshold_increment

            # 限制阈值范围在0-1之间
            threshold = max(0, min(1, threshold))

            return threshold

        except FileNotFoundError:
            print(f"规则文件未找到: {rules_file_path}")
            return initial_threshold
        except json.JSONDecodeError:
            print(f"规则文件格式错误: {rules_file_path}")
            return initial_threshold
        except Exception as e:
            print(f"检查规则时发生错误: {e}")
            return initial_threshold
