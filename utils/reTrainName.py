"""
# @Time    : 2022/12/13 13:26
# @File    : reTrainName.py
# @Author  : rezheaiba
"""
import os


def rename(path: str):
    for file in os.listdir(path):
        os.chdir(path)
        os.rename(file, file.replace('.jpg', '_leftImg8bit.png'))


if __name__ == '__main__':
    path = r'D:\Dataset\cityscapes\leftImg8bit\train\bdd10k_val'
    rename(path)
