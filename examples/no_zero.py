import os
import openai
from openai import OpenAI
import pandas as pd
import time
from alignment_check import AlignmentCheck

bailian_client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-2aa1148e2a794caaa3a10daf841cb3d3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0  # 增加超时时间
)

def get_completion(prompt, role_prompt, model, client, stream=False):
    """
    根据给定输入，调用模型进行回复
    @prompt: 对应的提示词
    @model: 调用的模型
    @stream: 是否启用流式输出
    @return: 返回答案
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": role_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            stream=stream  # 启用流式输出
        )

        if stream:
            # 处理流式响应
            full_response = ""
            for chunk in chat_completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)  # 实时打印输出
            print()  # 换行
            return full_response
        else:
            # 处理非流式响应
            return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"请求出错: {e}")
        raise e

if __name__ == '__main__':
    start_time = time.time()
    # 读取数据
    df = pd.read_excel("../resources/data/测试结果/CatQA+JailBench_seed+JailBench+normal/zero_test1.xlsx")
    # 读取提示词
    prompts = df[3550:3551]['text'].tolist()
    print(len(prompts))
    answers = []
    num = 1

    for prompt in prompts:
        print(f"第{num}条提示词: {prompt[:50]}...")  # 显示部分提示词内容
        try:
            # 使用流式输出处理大模型响应
            agent_answer = get_completion(prompt, "", "qwen2-72b-instruct", bailian_client, stream=False)
            answers.append(agent_answer)
            print(f"回答: {agent_answer[:100]}...")  # 显示部分回答内容
        except openai.BadRequestError as e:
            error_msg = "异常输入，有害内容"
            answers.append(error_msg)
            print(f"错误: {error_msg}")
        except openai.APIConnectionError as e:
            error_msg = "连接错误"
            answers.append(error_msg)
            print(f"连接错误: {e}")
        except openai.APITimeoutError as e:
            error_msg = "请求超时"
            answers.append(error_msg)
            print(f"超时错误: {e}")
        except Exception as e:
            error_msg = "未知错误"
            answers.append(error_msg)
            print(f"其他错误: {e}")
        finally:
            num += 1

    answers_df = pd.DataFrame({"answer": answers})
    answers_df.to_excel("../resources/data/测试结果/CatQA+JailBench_seed+JailBench+normal/qwen2-72b-instruct-no_zero3.xlsx", index=False)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} seconds")
