import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 必须输出 True
print(torch.cuda.get_device_name(0))