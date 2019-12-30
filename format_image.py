import os
import numpy as np
import cv2
from mypath import Path

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(file_path,image):
    cv2.imencode(file_path[-4:], image)[1].tofile(file_path)

if __name__ == "__main__":
    roi = Path.roi()
    x1 = roi[0]
    y1 = roi[1]
    x2 = roi[2]
    y2 = roi[3]
    lists = Path.crop_path()
    file_list = []
    file_index = []
    for data_dir in lists:
        temp_list = os.listdir(data_dir)
        file_list += temp_list
        file_index += [data_dir]*len(temp_list)

    for data_dir,file_name in zip(file_index,file_list):
        image_name = data_dir+file_name
        mask_name = data_dir+'mask/'+file_name[:-4]+'.png'
        if not os.path.isfile(image_name):
            continue
        temp1 = cv_imread(image_name)
        if temp1 is None:
            continue
        if temp1.shape[0]>=y2 and temp1.shape[1]>=x2:
            temp1 = temp1[y1:y2,x1:x2,:]
            cv_imwrite(image_name,temp1)
        if not os.access(mask_name, os.R_OK):
            continue
        temp2 = cv_imread(mask_name)
        if temp2.shape[0]>=y2 and temp2.shape[1]>=x2:
            temp2 = temp2[y1:y2,x1:x2]
            cv_imwrite(mask_name,temp2)