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

class get_file(Dataset):
    NUM_CLASSES = 2
    
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('water'),
                 split='train'):
        super().__init__()
        self.batch_num = 1
        if split is 'train':
            self.batch_num = 1
        self.split = split
        self.list = []
        self.index = []
        for dir_file in base_dir:
            lists = os.listdir(dir_file)
            for file_name in lists:
                temp_file_name = file_name[:-4] +".png"
                if os.access(dir_file+"mask/"+temp_file_name, os.R_OK):
                    self.list = self.list + [file_name]
                    self.index = self.index + [dir_file]
        self.args = args

    def __getitem__(self, index):
        index = math.floor(index/self.batch_num)
        _img, _target = self._make_img_pair(index)
        sample = {'image': _img, 'label': _target}
        
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_pair(self, index):
        img_dir = self.index[index]
        mask_dir = self.index[index]+'mask/'
        _img = cv2.imread(img_dir+ self.list[index])
        _target = cv2.imread(mask_dir+ self.list[index][:-4]+'.png')
        if _target.shape[2]>1:
            _target  = cv2.cvtColor(_target, cv2.COLOR_BGR2GRAY)
        _target[_target<100] = 0
        _target[_target>100] = 1
        
        return Image.fromarray(cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)), Image.fromarray(_target)


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
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
        return self.batch_num*len(self.list)
        # return len(self.list)

