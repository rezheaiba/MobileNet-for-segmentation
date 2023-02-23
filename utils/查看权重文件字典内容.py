"""
# @Time    : 2022/12/3 23:42
# @File    : 查看权重文件字典内容.py
# @Author  : rezheaiba
"""
import torch

if __name__ == '__main__':
    weight = torch.load('model_312.pth')
    # weight = torch.load('MobileNetV2_final_best.pth')
    print(weight['model'])
    # print(weight)
