import torch
from torch import nn, Tensor

from src.lraspp_lh import LRASPP
# from src.lraspp import LRASPP


class my_modle(LRASPP):
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)['out']


class out_model(LRASPP):
    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)['out']
        # prediction = output.argmax(1).squeeze(0)  # 16by16
        prediction = torch.max(output.squeeze(0), dim=0)[1]

        # probability = torch.softmax(output, dim=1).squeeze(0)
        probability = torch.softmax(output.squeeze(0), dim=0)

        probability_maxValue = torch.max(probability, dim=0)[0]
        # mutil = torch.tensor(100, dtype=torch.float)
        # probability_maxValue *= mutil
        # # probability_maxValue = probability_maxValue.type(torch.int64)
        # probability_maxValue = probability_maxValue.type_as(prediction)

        # pp = torch.cat((prediction.unsqueeze(0), probability_maxValue.unsqueeze(0)), dim=0)
        # return prediction
        return probability  # 19,16,16

# model = out_model(num_classes=19, backbone='mobilenet_v3_small')
# input = torch.randn(1, 3, 256, 256)
# output = model(input)
# print(output.shape)
