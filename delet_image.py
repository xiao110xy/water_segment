import os
import numpy as np
from mypath import Path
import time 

if __name__ == "__main__":
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
