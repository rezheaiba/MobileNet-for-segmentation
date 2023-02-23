"""
# @Time     : 2022/9/14 15:36
# @File     : img2ascll.py
# @Author   : rezheaiba
"""
import numpy as np
from PIL import Image

png_file = r"ADE_train_00000005.png"
img = Image.open(png_file)
# img = img.resize((int(img.size[0] * 0.2), int(img.size[1] * 0.2)))
img = img.resize((int(img.size[0]), int(img.size[1])))
print(img.size)

img_arr = np.asarray(img)
# 统计图像中的像素点数值，类别
label = np.unique(img_arr)
print(label)

height, width = img_arr.shape
print(img_arr.shape)
print(height, width)

img2code = ''
for i in range(height):
    for j in range(width):
        # pixel = img.getpixel((j, i))
        # img2code += ascii(pixel)
        img2code += ascii(img_arr[i][j])
    img2code += '\n'

fo = open('ascll.txt', 'w')
fo.write(img2code)
fo.close()
