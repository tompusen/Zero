import re
from regex_scanner import RegexScanner
if __name__ == '__main__':
    # 指定YAML文件所在目录，修改为实际路径
    yaml_directory = "../resources/rules/code_safe_regex/"
    # ["你好, zero", "hashlib.md5('data')"]
    prompts = "hashlib.md5('data')"
    regex_scanner = RegexScanner()
    # 加载规则
    rules = regex_scanner.load_yaml_rules(yaml_directory)
    # 匹配规则
    matches = regex_scanner.check_output_against_rules(prompts, rules)
    print(f"Loaded {len(rules)} rules from YAML files.")