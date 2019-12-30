import os
import numpy as np
from modeling.deeplab import *
import torch
import torchvision
import cv2
from mypath import Path
import time 
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torchvision import transforms
from dataloaders import custom_transforms as tr


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),1)
    return cv_img
def cv_imwrite(file_path,image):
    cv2.imencode(file_path[-4:], image)[1].tofile(file_path)

def main():
    roi = Path.roi()
    x1 = roi[0]
    y1 = roi[1]
    x2 = roi[2]
    y2 = roi[3]
    torch.backends.cudnn.benchmark=True
    # Define network
    model = DeepLab(num_classes=2,
                    backbone='drn',
                    output_stride = 8,
                    sync_bn=True,
                    freeze_bn=True)

    if not os.path.isfile(Path.test_model_path()):
        print('there is no *.pth.tar file')
        return
    print(Path.test_model_path())

    checkpoint = torch.load(Path.test_model_path())
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint


    # createVar = locals()
    # for i in range(40):
    #     start = time.time()
    #     model = DeepLab(num_classes=2,
    #                     backbone='drn',
    #                     output_stride = 8,
    #                     sync_bn=True,
    #                     freeze_bn=True)

    #     if not os.path.isfile(Path.test_model_path()):
    #         print('there is no *.pth.tar file')
    #         return
    #     print(Path.test_model_path())

    #     createVar['model'+str(i)] = model
    #     checkpoint = torch.load(Path.test_model_path())
    #     createVar['model'+str(i)].load_state_dict(checkpoint['state_dict'])
    #     del checkpoint
    #     print(time.time()-start)



    start = time.time()
    #checkpoint = torch.load('D:/Desktop/water/run/water/deeplab-drn/model_best.pth.tar')
    if torch.cuda.is_available():
        model.cuda()
        model.eval()
    print(time.time()-start)

    lists = Path.test_image_path()
    file_list = []
    file_index = []
    for data_dir in lists:
        temp_list = os.listdir(data_dir)
        file_list += temp_list
        file_index += [data_dir]*len(temp_list)
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    
    with torch.no_grad():
        # 添加文件夹判断
        start = time.time()
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
            # if os.access(save_name2, os.R_OK):
            #     continue
            im = cv_imread(image_name)

            if im is None:
                continue

            if im.shape[0]>=y2 and im.shape[1]>=x2:
                im = im[y1:y2,x1:x2,:]

            # im =cv2.resize(im,(512,512))
            # if im.shape[2]==3:
            #     im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            image = im.copy()

            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

            im = (im/255.0-mean)/std
            # numpy (H,W,C) --> Tensor (C,H,W)
            im=im.transpose(2,0,1)
            im=im[np.newaxis,...]

            input = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                input = input.cuda()

            with torch.no_grad(): 
                output = model(input)
                # torch.cuda.synchronize()
                pre1 = torch.max(output,1)[1][0]
                pre = pre1.cpu().numpy()


                pre[pre>0.5] = 255
                pre[pre<0.5] = 50
                image1 = image.copy()
                for i in [0,1] :
                    g = image1[:,:,i]
                    g[pre>100] = 255
                    image1[:,:,i] = g
                for i in [2]:
                    g = image1[:,:,i]
                    g[pre>100] = g[pre>100]/2
                    image1[:,:,i] = g

                # plt.imshow(pre)
                # plt.show()
                cv_imwrite(save_name1+'.png',image1)
                # image1 = image.copy()
                # for i in [0,1,2] :wei
                #     g[pre<100] = 0
                #     image1[:,:,i] = g
                # cv_imwrite(save_name1+'_1.png',image1)
                # image2 = image.copy()
                # for i in [0,1,2] :
                #     g = image2[:,:,i]
                #     g[pre>100] = 0
                #     image2[:,:,i] = g
                # cv_imwrite(save_name1+'_2.png',image2)

                cv_imwrite(save_name2,pre)
                print(file_name)
        print(time.time()-start)

if __name__ == "__main__":
   main()
