"""
# @Time    : 2022/12/8 11:33
# @File    : 19to6.py
# @Author  : rezheaiba
"""
import os

import numpy as np
from PIL import Image
from cv2 import cv2


def convert(imgpath: str, path: str):
    img_list = os.listdir(imgpath)

    for (idx, file) in enumerate(os.listdir(path)):
        src_name = os.path.join(path, file)
        dst_name = os.path.join(path, img_list[idx].split('.')[0] + '_gtFine_labelTrainIds_6.png')
        os.rename(src_name, dst_name)
        label = np.asarray(Image.open(dst_name).convert('L'))
        # road
        label[label == 1] = 0
        # building
        label[label == 2] = 1
        label[label == 3] = 1
        label[label == 4] = 1
        label[label == 5] = 1
        label[label == 6] = 1
        label[label == 7] = 1
        # vegetation
        label[label == 8] = 2
        label[label == 9] = 2
        # sky
        label[label == 10] = 3
        # human
        label[label == 11] = 4
        label[label == 12] = 4
        # car
        label[label == 13] = 5
        label[label == 14] = 5
        label[label == 15] = 5
        label[label == 16] = 5
        label[label == 17] = 5
        label[label == 18] = 5
        cv2.imwrite(dst_name, label)


if __name__ == '__main__':
    img_path = r'D:\Dataset\cityscapes\leftImg8bit\train\bdd10k_val'
    path = r'D:\Dataset\cityscapes\gtFine\train\bdd10k_val'
    # convert(img_path, path)