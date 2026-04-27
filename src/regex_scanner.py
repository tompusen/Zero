import json
import re
from ahocorasick import Automaton  # 需要安装 pyahocorasick 库


class RegexScanner:
    def __init__(self):
        self.automaton = Automaton()  # 初始化 Aho-Corasick 自动机
        self.rules_loaded = False  # 标记规则是否已加载

    def load_rules(self, rules_file_path="../resources/rules/综合规则库/综合规则.json"):
        """
        加载规则文件并将正则表达式添加到自动机中

        :param rules_file_path: 规则文件路径
        """
        try:
            with open(rules_file_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)

            # 将所有正则表达式添加到自动机中
            for rule in rules:
                if 'Regex' in rule and 'Description' in rule:
                    try:
                        # 直接将规则文本作为关键词添加到自动机
                        self.automaton.add_word(rule['Regex'], rule['Description'])
                    except re.error as e:
                        print(f"正则表达式编译失败: {rule['Regex']}, 错误: {e}")
                        continue

            # 构建自动机
            self.automaton.make_automaton()
            self.rules_loaded = True

        except FileNotFoundError:
            print(f"规则文件未找到: {rules_file_path}")
        except json.JSONDecodeError:
            print(f"规则文件格式错误: {rules_file_path}")
        except Exception as e:
            print(f"加载规则时发生错误: {e}")

    def alone_check_text(self, input_string):
        """
        检查输入字符串是否匹配规则库中的规则

        :param input_string: 待检查的输入字符串
        :return: 如果匹配到规则则返回Description内容，否则返回原字符串
        """
        if not self.rules_loaded:
            raise RuntimeError("规则尚未加载，请先调用 load_rules 方法")

        # 使用自动机进行快速匹配
        for _, description in self.automaton.iter(input_string):
            return description  # 返回第一个匹配的规则描述

        return input_string  # 无匹配时返回原字符串

    def union_check_text(self, input_string, initial_threshold=0.5, threshold_increment=0.15, threshold_decrement=0.15):
        """
        检查输入字符串是否匹配规则库中的规则，并根据匹配情况调整阈值

        :param input_string: 待检查的输入字符串
        :param initial_threshold: 初始阈值，默认为0.5
        :param threshold_increment: 匹配失败时阈值增加量
        :param threshold_decrement: 匹配成功时阈值减少量
        :return: 调整后的阈值
        """
        if not self.rules_loaded:
            raise RuntimeError("规则尚未加载，请先调用 load_rules 方法")

        threshold = initial_threshold
        matched = False

        # 使用自动机进行快速匹配
        for _, _ in self.automaton.iter(input_string):
            threshold -= threshold_decrement
            matched = True
            break  # 只需检测是否存在匹配即可

        # 如果没有匹配到任何规则，调整阈值
        if not matched:
            threshold += threshold_increment

        # 限制阈值范围在0-1之间
        threshold = max(0, min(1, threshold))
        return threshold


if __name__ == '__main__':
    regex_scanner = RegexScanner()
    regex_scanner.load_rules()
    print(regex_scanner.union_check_text("草拟犸"))