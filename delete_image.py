import os
import numpy as np
import cv2
from mypath import Path

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(file_path,image):
    cv2.imencode(file_path[-4:], image)[1].tofile(file_path)

def main():
    lists = Path.db_root_dir('water')
    file_list = []
    file_index = []
    for data_dir in lists:
        temp_list = os.listdir(data_dir)
        file_list += temp_list
        file_index += [data_dir]*len(temp_list)

    for data_dir,file_name in zip(file_index,file_list):
        image_name = data_dir+file_name
        save_name1 = data_dir+'mask/'+file_name[:-4]+'.png'
        if not os.path.isfile(image_name):
            continue
        if not os.access(save_name1, os.R_OK):
            print('delete {} !'.format(file_name))
            os.remove(image_name)

if __name__ == "__main__":
   main()


"""
    lists = Path.db_root_dir('water')
    file_list = []
    file_index = []
    for data_dir in lists:
        temp_list = os.listdir(data_dir)
        file_list += temp_list
        file_index += [data_dir]*len(temp_list)

    for data_dir,file_name in zip(file_index,file_list):
        image_name = data_dir+file_name
        flag = True
        exit_name = data_dir+'mask/'+file_name[:-4]+".png"
        save_name1 = data_dir+'result/'+file_name[:-4]+"_1.png"
        save_name2 = data_dir+'result/'+file_name[:-4]+'_2.png'
        if os.path.isfile(exit_name):
            flag = False
        if not os.path.isfile(save_name1):
            flag = False
        if not os.path.isfile(save_name2):
            flag = False
        if flag:
            os.remove(image_name)

"""
