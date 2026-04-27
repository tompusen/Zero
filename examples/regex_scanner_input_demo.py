import time

from regex_scanner import RegexScanner
import pandas as pd
if __name__ == '__main__':
    start_time = time.time()
    regex_scanner = RegexScanner()
    df = pd.read_excel("../resources/data/测试结果/CatQA+JailBench_seed+JailBench+normal/zero_test1.xlsx")
    inputs = df[:]['text'].tolist()
    regex_scanner.load_rules()
    for prompt in inputs:
        threshold = regex_scanner.union_check_text(prompt)
    normal_prompt = regex_scanner.union_check_text(inputs)
    print(f"未拦截: {len(normal_prompt)}, 已拦截: {len(inputs) - len(normal_prompt)}"
          f", 拦截率: {(len(inputs) - len(normal_prompt))/len(inputs)}")
    end_time = time.time()
    print(f"平均拦截时间: {(end_time - start_time)/len(inputs)}")
