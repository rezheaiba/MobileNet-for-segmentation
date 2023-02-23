import os
import torch
from model import my_modle, out_model

classes = 6
weights_path = "weights-cityscapes-6/model_376.pth"
# weights_path = './weights/lraspp_mobilenet_v3_large.pth'
assert os.path.exists(weights_path), f"weights {weights_path} not found."

model = out_model(num_classes=classes,
                  reduced_tail=True,
                  backbone="mobilenet_v3_small")
# model = my_modle(num_classes=classes + 1)

weights_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(weights_dict['model'])
model.to("cpu")

model.eval()
dummy_input = torch.randn(1, 3, 256, 256, device='cpu')
input_names = ['input']
output_names = ['output']
torch.onnx.export(model, dummy_input, "lraspp.onnx", opset_version=11,
                  input_names=input_names, output_names=output_names)