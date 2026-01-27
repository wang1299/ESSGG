CUDA_VISIBLE_DEVICES=0 

import torch
print("CUDA 可用:", torch.cuda.is_available())
print("设备数量:", torch.cuda.device_count())
print("当前设备:", torch.cuda.get_device_name(0))
print("可用显存:", torch.cuda.mem_get_info()[0] / 1024**2, "MB")

