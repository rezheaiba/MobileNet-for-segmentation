from collections import OrderedDict
from typing import Dict

import torch
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter

from mobilenetv3REPLACE import MobileNetV3


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
        backbone = MobileNetV3(dilated=True, reduced_tail=reduced_tail, dropout=dropout, arch=backbone, width_mult=0.5)
        backbone = backbone.features
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        low_pos = stage_indices[-3]
        mid_pos = stage_indices[-2]
        high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        low_channels = backbone[low_pos].out_channels
        mid_channels = backbone[mid_pos].out_channels
        high_channels = backbone[high_pos].out_channels

        # change name and out put
        backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(mid_pos): "mid",
                                                                    str(high_pos): "high"})

        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, mid_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        # out = F.interpolate(out,size=input.shape[-2:],mode="bilinear",align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result


class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, mid_channels: int, high_channels: int, num_classes: int,
                 inter_channels: int) -> None:
        super().__init__()
        self.cbrH = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        self.scaleH = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.cbrM = nn.Sequential(
            nn.Conv2d(mid_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scaleM = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.cbrL = nn.Sequential(
            nn.Conv2d(low_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scaleL = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_channels, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.low_classifier = nn.Conv2d(inter_channels, num_classes, 1)  # TO UPDATE
        self.mid_classifier = nn.Conv2d(inter_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        mid = input["mid"]
        high = input["high"]
        xH = self.cbrH(high)  # (1,128,16,16)
        sH = self.scaleH(high)
        xH = xH * sH

        xM = self.cbrM(mid)  # (1,128,16,16)
        sM = self.scaleM(mid)
        xM = xM * sM

        xL = self.cbrL(low)  # (1,128,16,16)
        sL = self.scaleL(low)
        xL = xL * sL
        # x = F.interpolate(x,size=low.shape[-2:],mode="bilinear",align_corners=False)

        return self.high_classifier(xH) + self.mid_classifier(xM) + self.low_classifier(xL)


def main():
    input = torch.randn([1, 3, 256, 256])
    model = LRASPP(num_classes=6, reduced_tail=True,
                   backbone="mobilenet_v3_small").cpu()
    model.eval()
    input = input.cpu()
    out = model(input)
    print(out['out'].shape)
    torch.save(model.state_dict(),"model_size.pth")

if __name__ == '__main__':
    main()
