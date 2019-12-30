class Path(object):
    @staticmethod
    def train_image_path():
        # 训练数据集所在文件夹
        # return ['E:/错误/带领/带领p3/']
        # return ['water/water_13/','water/water_14/','water/water_15/']
        return [r'water/p_1']
    @staticmethod
    def train_model_path():
        # 用于训练的模型文件路径
        return  ''
        # return  'checkpoint/experiment_0/checkpoint.pth.tar'
        
    # 后面不要加/或\
    @staticmethod
    def checkpoint_path():
        return 'checkpoint'
        # return 'D:/Desktop'

    @staticmethod
    def test_image_path():
        # 测试数据集所在文件夹
        # return ['E:/错误/菜嘴子/error/','E:/错误/带领/error/','E:/错误/秦皇岛/error/','E:/错误/三道营/error/']
        # return [r'water/p5/']
        return [r'water/p_1']
        # return [r'E:\water_segment\错误\秦皇岛\秦皇岛P1/']
        # return ['water/water_13/','water/water_14/','water/water_15/']


    @staticmethod
    def test_model_path():
        # 用于测试的模型文件路径
        # return  'E:/错误/best_model/带领/checkpoint.pth.tar'
        #return  'D:/Desktop/water_6/experiment_1/checkpoint.pth.tar'
        # return  'checkpoint/experiment_5/checkpoint.pth.tar'
        # return  'E:/water_segment/错误/best_model/三道营/checkpoint.pth.tar'
        return  'checkpoint/experiment_7/checkpoint.pth.tar'

    @staticmethod
    def convert_model_path():
        # 训练完成的模型文件路径
        # return  'checkpoint/experiment_15/checkpoint.pth.tar'
        return  'checkpoint/experiment_2/checkpoint.pth.tar'

    @staticmethod
    def convert_save_path():
        # 另存为pt文件
        return 'test.pt'
    

    @staticmethod
    def train_batch_size():
        # 训练批次
        return 4

    @staticmethod
    def test_batch_size():
        #测试批次
        return 4
    

    @staticmethod
    def crop_path():
        # 裁剪的文件夹
        return  ['water/water_12/']

    @staticmethod
    def roi():
        #测试批次
        # x1 y1 x2 y2
        return [0,0,12000,8000]# P3 
        # return [527,29,783,679]# P3 
        # return [527,29,783,679]
