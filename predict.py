import os
import time
import json

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from model import my_modle, out_model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 6
    weights_path = r"weights-cityscapes-6/model_376.pth"
    # weights_path = r"weights/cityscapes_169.pth"
    img_path = r"D:\Dataset\cityscapes\leftImg8bit\test\berlin\berlin_000014_000019_leftImg8bit.png"
    img_path = r"D:\Dataset\data\final-450\train\outdoor\sunny_00036.jpg"
    palette_path = "palette_cityscapes6.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    # rb: 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
    with open(palette_path, "rb") as f:
        pallette_dic = json.load(f)
        pallette = []
        for v in pallette_dic.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # creat model
    model = my_modle(num_classes=classes,
                     reduced_tail=True,
                     backbone="mobilenet_v3_small")

    # load weights
    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict['model'])
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transfrom = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])
    img = data_transfrom(original_img)

    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()

    with torch.no_grad():
        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print(f"inference time: {(t_end - t_start)}")

        #  '''
        #  后处理，量化时需要
        #  probability = torch.softmax(output, dim=1).squeeze(0):输出的是19*16*16的概率图
        #  probability_maxValue = torch.max(x, dim=1)[0]:输出16*16的最大概率图
        #  '''
        #  prediction = output.argmax(1).squeeze(0)  # 16by16
        prediction = torch.max(output.squeeze(0), dim=0)[1]  # 16by16
        print('prediction_tensor_type:\n', prediction)
        #
        # 输出一下概率
        probability = torch.softmax(output, dim=1).squeeze(0)
        probability_maxValue = torch.max(probability, dim=0)[0].data

        # 把概率转变成long类型
        # mutil = torch.tensor(100)
        # probability_maxValue *= mutil
        # # probability_maxValue = probability_maxValue.type(torch.int64)
        # probability_maxValue = probability_maxValue.type_as(prediction)

        # print('probability_maxValue:\n', probability_maxValue)

        # 量化后的模型最好只输出一个tensor，所以把这两个合并一下
        pp = torch.cat((prediction.unsqueeze(0), probability_maxValue.unsqueeze(0)), dim=0)
        # print(pp)
        # print(probability.shape)
        # print(probability)
        '''
        # probability = torch.permute(probability, (1,2,0))
        # probability = output
        print(probability.shape)
        for i in range(probability.shape[0]):
            for j in range(probability.shape[1]):
                for k in range(probability.shape[2]):
                    print(str(i), str(j), str(k), probability[i, j, k])
                print('\n')
        '''
    # 转8bit的numpy
    prediction = prediction.to('cpu').numpy().astype(np.uint8)
    # print('prediction_numpy_type:', prediction)  # 打印numpy查看一下类别

    # 转PIL
    mask = Image.fromarray(prediction)
    # pallette是一个list
    mask.putpalette(pallette)
    mask.save("cityscapes_result.png")


if __name__ == '__main__':
    main()
