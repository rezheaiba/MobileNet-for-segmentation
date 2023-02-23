"""
# @Time    : 2022/12/20 15:35
# @File    : FLOPS.py
# @Author  : rezheaiba
"""
import torch
from torchvision.models import *
from src.mobilenetv3REPLACE import MobileNetV3
from thop import profile
from thop import clever_format

model = MobileNetV3(num_classes=7, arch='mobilenet_v3_small',width_mult=0.25)
model = resnet50(pretrained=False)

input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input, ))
print('parmas:{}, flops:{}'.format(params, flops))
print(
        "%s | %.2f M | %.2f G" % ('mobilenet_v3', params / (1000 ** 2), flops / (1000 ** 3))
    )
flops, params = clever_format([flops, params], '%3.f')
print('parmas:{}, flops:{}'.format(params, flops))


'''mobilenet_v3 | 0.12 M | 0.01 G
parmas:119K, flops: 14M'''