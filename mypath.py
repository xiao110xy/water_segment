class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return 'E:/data/coco/'
        elif dataset == 'water':
            # return ['D:/Desktop/湛江/湛江港/','E:/water_segment/water_1/']
            # return ['D:/Desktop/湛江/湛江港/image_P3/',
            #         'D:/Desktop/湛江/湛江港/image_P4/',
            #         'D:/Desktop/湛江/湛江港/image_P1/',
            #         'D:/Desktop/湛江/湛江港/image_P2/']
            return ['E:/water_segment/water_3/']
            # return ['D:/Desktop/湛江/湛江港/image_P3/','D:/Desktop/湛江/湛江港/image_P2/']
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
