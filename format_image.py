import os
import numpy as np
import cv2

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(file_path,image):
    cv2.imencode(file_path[-4:], image)[1].tofile(file_path)

if __name__ == "__main__":
    x1 = 550
    y1 = 30
    x2 = 770
    y2 = 530
    lists = ['water/water_6/']
    file_list = []
    file_index = []
    for data_dir in lists:
        temp_list = os.listdir(data_dir)
        file_list += temp_list
        file_index += [data_dir]*len(temp_list)

    for data_dir,file_name in zip(file_index,file_list):
        image_name = data_dir+file_name
        begin_mask_name = data_dir+'begin_mask/'+file_name[:-4]+'.png'
        mask_name = data_dir+'mask/'+file_name[:-4]+'.png'
        if not os.path.isfile(image_name):
            continue
        if not os.access(begin_mask_name, os.R_OK):
            continue
        if not os.access(mask_name, os.R_OK):
            continue
        temp1 = cv_imread(image_name)
        temp2 = cv_imread(begin_mask_name)
        temp3 = cv_imread(mask_name)
        temp1 = temp1[y1:y2,x1:x2,:]
        temp2 = temp2[y1:y2,x1:x2]
        temp3 = temp3[y1:y2,x1:x2]
        cv_imwrite(image_name,temp1)
        cv_imwrite(begin_mask_name,temp2)
        cv_imwrite(mask_name,temp3)