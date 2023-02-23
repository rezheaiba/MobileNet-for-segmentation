from functools import partial
from collections import OrderedDict
from typing import Any, Dict, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from lraspp.mobilenetv3 import MobileNetV3


# from mobilenetv3REPLACE import MobileNetV3


class LRASPP(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            inter_channels: int = 128,
            reduced_tail: bool = False,
            dropout: float = 0.2,
            backbone: str = "mobilenet_v3_large"
    ) -> None:
        super().__init__()
        backbone = MobileNetV3(dilated=True, reduced_tail=reduced_tail, dropout=dropout, arch=backbone)
        backbone = backbone.features
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        c1 = stage_indices[-5]  # use C2 here which has output_stride = 8
        c2 = stage_indices[-4]  # use C2 here which has output_stride = 8
        c3 = stage_indices[-3]  # use C2 here which has output_stride = 8
        c4 = stage_indices[-2]  # use C2 here which has output_stride = 8
        c5 = stage_indices[-1]  # use C5 which has output_stride = 16
        c1_channels = backbone[c1].out_channels
        c2_channels = backbone[c2].out_channels
        c3_channels = backbone[c3].out_channels
        c4_channels = backbone[c4].out_channels
        c5_channels = backbone[c5].out_channels

        # change name and out put
        backbone = IntermediateLayerGetter(backbone,
                                           return_layers={str(c1): "c1", str(c2): "c2", str(c3): "c3", str(c4): "c4",
                                                          str(c5): "c5"})

        self.backbone = backbone
        self.classifier = LRASPPHead(c1_channels, c2_channels, c3_channels, c4_channels, c5_channels, num_classes)

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)

        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result


class LRASPPHead(nn.Module):
    def __init__(self, c1_channels: int, c2_channels: int, c3_channels: int, c4_channels: int, c5_channels: int,
                 num_classes: int) -> None:
        super().__init__()
        # self.att = Attention_block(1, 1, 1)

        self.cbr5 = nn.Sequential(
            nn.Conv2d(c5_channels, c4_channels, 1, bias=False),
            nn.BatchNorm2d(c4_channels),
            # nn.ReLU(inplace=True),
        )
        self.scale5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4_channels, c4_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.cbr4 = nn.Sequential(
            nn.Conv2d(c4_channels, c3_channels, 1, bias=False),
            nn.BatchNorm2d(c3_channels),
            # nn.ReLU(inplace=True),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c3_channels, c3_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.cbr3 = nn.Sequential(
            nn.Conv2d(c3_channels, c2_channels, 1, bias=False),
            nn.BatchNorm2d(c2_channels),
            # nn.ReLU(inplace=True),
        )
        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2_channels, c2_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(c2_channels, c1_channels, 1, bias=False),
            nn.BatchNorm2d(c1_channels),
            # nn.ReLU(inplace=True),
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1_channels, c1_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        self.cb4 = nn.Sequential(
            nn.Conv2d(c4_channels, c1_channels, 1, bias=False),
            nn.BatchNorm2d(c1_channels),
        )

        self.cb3 = nn.Sequential(
            nn.Conv2d(c3_channels, c1_channels, 1, bias=False),
            nn.BatchNorm2d(c1_channels),
        )

        self.cb2 = nn.Sequential(
            nn.Conv2d(c2_channels, c1_channels, 1, bias=False),
            nn.BatchNorm2d(c1_channels),
        )

        self.cb1 = nn.Sequential(
            nn.Conv2d(c1_channels, c1_channels, 1, bias=False),
            nn.BatchNorm2d(c1_channels),
        )

        self.classifier = nn.Conv2d(c1_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        c1 = input["c1"]  # 16,56,56
        c2 = input["c2"]  # 24,28,28
        c3 = input["c3"]  # 40,14,14
        c4 = input["c4"]  # 48,14,14
        c5 = input["c5"]  # 288,14,14

        c5 = self.cbr5(c5)
        s5 = self.scale5(c5)
        c5 = c5 * s5
        # c5 = F.interpolate(c5, size=c4.shape[-2:], mode="bilinear", align_corners=False)

        x4 = self.relu(torch.add(c5, c4))
        # x4 = torch.add(c5, c4)

        c4 = self.cbr4(c4)
        s4 = self.scale4(c4)
        c4 = c4 * s4

        # c4 = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)

        x3 = self.relu(torch.add(c4, c3))
        # x3 = torch.add(c4, c3)
        c3 = self.cbr3(c3)
        s3 = self.scale3(c3)
        c3 = c3 * s3

        c3 = F.interpolate(c3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        x2 = self.relu(torch.add(c3, c2))
        # x2 =torch.add(c3, c2)
        c2 = self.cbr2(c2)
        s2 = self.scale2(c2)
        c2 = c2 * s2

        c2 = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.relu(torch.add(c2, c1))
        # x1 = torch.add(c2, c1)

        x4 = self.cb4(x4)
        x4 = F.interpolate(x4, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        x3 = self.cb3(x3)
        x3 = F.interpolate(x3, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        x2 = self.cb2(x2)
        x2 = F.interpolate(x2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        x1 = self.cb1(x1)

        x = torch.add(x1, x2)
        x = torch.add(x, x3)
        x = torch.add(x, x4)
        x = self.relu(x)

        return self.classifier(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  # 卷积不改变尺寸
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  # channel数都为F_int
        psi = self.psi(psi)

        return x * psi


def main():
    input = torch.randn([1, 1, 224, 224])
    model = LRASPP(num_classes=1, reduced_tail=True,
                   backbone="mobilenet_v3_small").cpu()
    model.eval()
    input = input.cpu()
    out = model(input)
    print(out['out'].shape)
    # torch.save(model.state_dict(),"model_size.pth")


# if __name__ == '__main__':
#     main()
