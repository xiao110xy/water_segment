class Path(object):
    @staticmethod
    def train_image_path():
        # 训练数据集所在文件夹
        return ['water/water_6/']
    @staticmethod
    def train_model_path():
        # 用于训练的模型文件路径
        return 'models/checkpoint.pth.tar'

    @staticmethod
    def test_image_path():
        # 测试数据集所在文件夹
        return ['water/water_6/']

    @staticmethod
    def test_model_path():
        # 用于测试的模型文件路径
        return 'models/checkpoint.pth.tar'

    @staticmethod
    def convert_model_path():
        # 训练完成的模型文件路径
        return 'models/checkpoint.pth.tar'

    @staticmethod
    def convert_save_path():
        # 另存为pt文件
        return 'test.pt'

    @staticmethod
    def train_batch_size():
        # 训练批次
        return 8

    @staticmethod
    def test_batch_size():
        #测试批次
        return 8