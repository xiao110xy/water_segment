import os
import numpy as np
from modeling.deeplab import *
import torch
import torchvision
from mypath import Path



def main():
    # Define network
    model = DeepLab(num_classes=2,
                    backbone='drn',
                    output_stride=8,
                    sync_bn=True,
                    freeze_bn=True)
    checkpoint = torch.load(Path.convert_model_path(),
        map_location=torch.device('cpu'))


    model.load_state_dict(checkpoint['state_dict'])

    model = model.eval()
    example = torch.rand(1, 3, 513, 513)
    model.forward(example)
    # model = torchvision.models.resnet18()
    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(Path.convert_save_path())
        print('convert file succeed')

if __name__ == "__main__":
   main()