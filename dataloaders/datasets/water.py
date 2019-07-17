import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import os
import glob
import cv2
import math
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile

last_name = ".png"
class get_file(Dataset):
    NUM_CLASSES = 2
    
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('water'),
                 split='train'):
        super().__init__()
        self.batch_num = 1
        if split is 'train':
            self.batch_num = 10
        self.split = split
        self.list = []
        self.index = []
        for dir_file in base_dir:
            lists = os.listdir(dir_file)
            for file_name in lists:
                temp_file_name = file_name[:-4] +last_name
                if os.access(dir_file+"mask/"+temp_file_name, os.R_OK):
                    self.list = self.list + [file_name]
                    self.index = self.index + [dir_file]
        self.image_mat = []
        self.mask_mat = []
        for idx in range(0, len(self.list)):
            image_name = dir_file + self.list[idx]
            mask_name = dir_file + "mask/" + self.list[idx][:-4] +last_name
            _img = cv2.imread(image_name)
            
            # _target = cv2.imread(mask_dir+ self.list[index][:-4]+'.png')
            _target = cv2.imread(mask_name)
            if _target.shape[2]>1:
                _target  = cv2.cvtColor(_target, cv2.COLOR_BGR2GRAY)
            _target[_target<100] = 0
            _target[_target>100] = 1
            self.image_mat.append(Image.fromarray(cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)))
            self.mask_mat.append(Image.fromarray(_target))
        self.args = args

    def __getitem__(self, index):
        index = math.floor(index/self.batch_num)
        _img, _target = self._make_img_pair_from_mat(index)
        sample = {'image': _img, 'label': _target}
        
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_pair_from_mat(self, index):
        return self.image_mat[index], self.mask_mat[index]

    def _make_img_pair(self, index):
        # temp_index = index % 16
        # if self.split == "train":
        #     index = int(index / 16)
        img_dir = self.index[index]
        mask_dir = self.index[index]+'mask/'
        _img = cv2.imread(img_dir+ self.list[index])
        # _target = cv2.imread(mask_dir+ self.list[index][:-4]+'.png')
        _target = cv2.imread(mask_dir+ self.list[index][:-4]+last_name)
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

