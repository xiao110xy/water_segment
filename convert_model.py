import os
import numpy as np
from modeling.deeplab import *
import torch
import torchvision



def main():
    # Define network
    model = DeepLab(num_classes=2,
                    backbone='mobilenet',
                    output_stride=8,
                    sync_bn=True,
                    freeze_bn=True)
    checkpoint = torch.load('D:/Desktop/water/run/water/deeplab-mobilenet/model_best.pth.tar',
        map_location=torch.device('cpu'))


    model.load_state_dict(checkpoint['state_dict'])

    model = model.eval()
    example = torch.rand(1, 3, 513, 513)
    model.forward(example)
    # model = torchvision.models.resnet18()
    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("E:/data/models/xy_4.pt")
        print('ok')

if __name__ == "__main__":
   main()