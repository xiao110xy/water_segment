import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.min_miou = 4
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def xy_Mean_Intersection_over_Union(self):
        return self.min_miou

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        for index in  range(gt_image.shape[0]):
            temp1 = gt_image[index][np.newaxis,...]
            temp2 = pre_image[index][np.newaxis,...]
            temp_confusion_matrix = self._generate_matrix(temp1, temp2)
            temp_MIoU = np.diag(temp_confusion_matrix) / (
                np.sum(temp_confusion_matrix, axis=1) + np.sum(temp_confusion_matrix, axis=0) -
                np.diag(temp_confusion_matrix))
            temp_MIoU = np.nanmean(temp_MIoU)
            if  temp_MIoU<self.min_miou:
                self.min_miou = temp_MIoU

        
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.min_miou = 4
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




