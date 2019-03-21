import os
import numpy as np
from modeling.deeplab import *
import torch
import torchvision
import cv2
from mypath import Path





def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(file_path,image):
    cv2.imencode(file_path[-4:], image)[1].tofile(file_path)
def main():
    # Define network
    model = DeepLab(num_classes=2,
                    backbone='mobilenet',
                    output_stride = 8,
                    sync_bn=True,
                    freeze_bn=True)
    # train_params = [{'params': model.get_1x_lr_params(), 'lr': 0.01},
    #                 {'params': model.get_10x_lr_params(), 'lr': 0.01 * 10}]

    # Define Optimizer
    # optimizer = torch.optim.SGD(train_params,
    #  momentum=0.9,weight_decay=0.005, nesterov =False)
    checkpoint = torch.load('D:/Desktop/water/run/water/deeplab-mobilenet/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # example = torch.rand(1,3,513,513)
    # traced_script_module = torch.jit.trace(model, example)
    # traced_script_module.save('E:/data/models/xy.pt')
    lists = Path.db_root_dir('water')
    file_list = []
    file_index = []
    for data_dir in lists:
        temp_list = os.listdir(data_dir)
        file_list += temp_list
        file_index += [data_dir]*len(temp_list)
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    model.cuda(1)
    model.eval()
    with torch.no_grad():
        # 添加文件夹判断
        for data_dir,file_name in zip(file_index,file_list):
            image_name = data_dir+file_name
            if not os.path.exists(data_dir+'result/'):
                os.makedirs(data_dir+'result/')
                print('mkdir result')
            if not os.path.exists(data_dir+'result_mask/'):
                os.makedirs(data_dir+'result_mask/')
                print('mkdir result_mask')
            save_name1 = data_dir+'result/'+file_name[:-4]
            save_name2 = data_dir+'result_mask/mask_'+file_name[:-4]+'.png'
            if not os.path.isfile(image_name):
                continue
            # if os.access(save_name1, os.R_OK):
            #     continue
            im = cv_imread(image_name)
            if im is None:
                continue

            # im =cv2.resize(im,(512,512))
            image = im.copy()
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            im = (im/255.0-mean)/std
            # numpy (H,W,C) --> Tensor (C,H,W)
            im=im.transpose(2,0,1)
            im=im[np.newaxis,...]
            input = torch.from_numpy(im).float().cuda(1)
            with torch.no_grad(): 
                output = model(input)
                pre = torch.max(output,1)[1][0]
                pre = pre.data.cpu().numpy()
                pre[pre>0.5] = 255
                pre[pre<0.5] = 50

                image1 = image.copy()
                for i in [0,1,2] :
                    g = image1[:,:,i]
                    g[pre<100] = 0
                    image1[:,:,i] = g
                cv_imwrite(save_name1+'_1.png',image1)
                image2 = image.copy()
                for i in [0,1,2] :
                    g = image2[:,:,i]
                    g[pre>100] = 0
                    image2[:,:,i] = g
                cv_imwrite(save_name1+'_2.png',image2)

                cv_imwrite(save_name2,pre)
                print(file_name)

if __name__ == "__main__":
   main()
