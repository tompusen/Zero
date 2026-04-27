import time
from regex_scanner import RegexScanner
from prompt_guard import PromptGuard
import pandas as pd

if __name__ == '__main__':
    start_time = time.time()
    model_path = "../resources/model/5000_zero"
    tokenizer_path = "../resources/model/5000_zero"
    regex_scanner = RegexScanner()
    regex_scanner.load_rules()
    prompt_guard = PromptGuard(tokenizer_path, model_path)
    df = pd.read_excel("../resources/data/test_data.xlsx")

    prompts = df[:]['text'].tolist()
    results = []
    for index, prompt in enumerate(prompts):
        print(f"进度{index+1}/{len(prompts)}\n提示词: {prompt}\n")
        threshold = regex_scanner.union_check_text(prompt, 0.15)
        pre_result = prompt_guard.block_prompt(prompt, threshold)
        results.append(pre_result)
    df["pre"] = results
    df.to_excel("new_zero_test.xlsx", index=False)
    end_time = time.time()
    print("程序运行时长: ", end_time - start_time)
