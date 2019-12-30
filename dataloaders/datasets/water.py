import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import os
import glob
import cv2
import math
import torch
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
import re
import matplotlib.pyplot as plt
last_name = ".png"


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

class get_file(Dataset):
    NUM_CLASSES = 2
    
    def __init__(self,
                 args,
                 base_dir = '',
                 split='train'):
        super().__init__()
        if split is 'train':
            base_dir = Path.train_image_path()
        else:
            base_dir = Path.test_image_path()
        self.batch_num = 1
        # if split is 'train':
        #     self.batch_num = 10
        self.split = split
        self.list = []
        self.index = []
        self.map_weight_mat = {}
        for dir_file in base_dir:
            lists = os.listdir(dir_file)
            for file_name in lists:
                temp_file_name = file_name[:-4] +last_name
                if os.access(dir_file+"mask/"+temp_file_name, os.R_OK) and os.access(dir_file+file_name, os.R_OK):
                    self.map_weight_mat[dir_file+file_name[:-4]] = 1
                    self.list = self.list + [file_name]
                    self.index = self.index + [dir_file]
            if os.access(dir_file+'weight.txt', os.R_OK):
                with open(dir_file+'weight.txt', 'r', encoding='UTF-8') as file:
                    print("weight used")
                    lines = file.readlines()    # 接收数据
                    for line in lines:     # 遍历数据
                        line = line.replace("\r","").replace("\n","")
                        data = line.split(',')

                        if len(data) is 2 and dir_file+data[0] in self.map_weight_mat:
                            self.map_weight_mat[dir_file+data[0]] = int(data[1])
        self.image_mat = []
        self.mask_mat = []
        
        for idx in range(0, len(self.list)):
            image_name =  self.index[idx] + self.list[idx]
            mask_name = self.index[idx] + "mask/" + self.list[idx][:-4] +last_name
            _img = cv_imread(image_name)

            # _target = cv2.imread(mask_dir+ self.list[index][:-4]+'.png')
            _target = cv_imread(mask_name)


            if len(_target.shape)>2:
                if _target.shape[2]>1:
                    _target  = cv2.cvtColor(_target, cv2.COLOR_BGR2GRAY)
            _target[_target<100] = 0
            _target[_target>100] = 1
            self.image_mat.append(Image.fromarray(cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)))
            self.mask_mat.append(Image.fromarray(_target))
        self.args = args

    def __getitem__(self, index):
        index = math.floor(index/self.batch_num)
        _img, _target,_weight = self._make_img_pair_from_mat(index)
        sample = {'image': _img, 'label': _target}
        
        temp ={}
        #temp['weight'] =  tr.ToTensor(_weight)
        if self.split == "train":
            temp = self.transform_tr(sample)
        elif self.split == 'val':
            temp = self.transform_val(sample)
        temp['weight'] = torch.FloatTensor([[_weight]])

        # _img = temp['image']
        # _target = temp['label']
        # image = np.array(_img)
        # image = np.transpose(image,(1,2,0))
        # pre = np.array(_target)
        # image1 = image.copy()
        # for i in [0] :
        #     g = image1[:,:,i]
        #     g[pre>0.5] = 255
        #     image1[:,:,i] = g
        # for i in [1,2]:
        #     g = image1[:,:,i]
        #     g[pre>0.5] = 255
        #     image1[:,:,i] = g
        # plt.imshow(image1)
        # plt.show()
        
        return temp


    def _make_img_pair_from_mat(self, index):
        name =  self.index[index]+self.list[index][:-4]
        return self.image_mat[index], self.mask_mat[index],self.map_weight_mat[name]

    def _make_img_pair(self, index):
        # temp_index = index % 16
        # if self.split == "train":
        #     index = int(index / 16)
        img_dir = self.index[index]
        mask_dir = self.index[index]+'mask/'
        _img = cv_imread(img_dir+ self.list[index])
        # _target = cv2.imread(mask_dir+ self.list[index][:-4]+'.png')
        _target = cv_imread(mask_dir+ self.list[index][:-4]+last_name)
        if _target.shape[2]>1:
            _target  = cv2.cvtColor(_target, cv2.COLOR_BGR2GRAY)
        _target[_target<100] = 0
        _target[_target>100] = 1
        # temp1 = temp_index % 4
        # temp2 = int(temp_index/4)
        # if self.split == "train":
        #     _img = _img[temp1*512:(temp1+1)*512,temp2*512:(temp2+1)*512,:]
        #     _target = _target[temp1*512:(temp1+1)*512,temp2*512:(temp2+1)*512]
        return Image.fromarray(cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)), Image.fromarray(_target)


    def transform_tr(self, sample):
        # if (sample['image'].width>self.args.base_size*2) and (sample['image'].height>self.args.base_size*2):
        #     composed_transforms = transforms.Compose([
        #         tr.RandomHorizontalFlip(),
        #         tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
        #         tr.RandomGaussianBlur(),
        #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #         tr.ToTensor()])
        # else:
        #     composed_transforms = transforms.Compose([
        #         # tr.FixScaleCrop(crop_size=self.args.crop_size),
        #         tr.RandomHorizontalFlip(),
        #         tr.RandomGaussianBlur(),
        #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #         tr.ToTensor()])
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        # if self.split == "train":
        #     return self.batch_num*len(self.list)*16
        # else:
            return self.batch_num*len(self.list)    
        # return len(self.list)

