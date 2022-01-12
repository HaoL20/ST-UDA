import numpy as np

np.seterr(divide='ignore', invalid='ignore')

name_classes = [
    'road',             # 0
    'sidewalk',         # 1
    'building',         # 2
    'wall',             # 3
    'fence',            # 4
    'pole',             # 5
    'trafflight',       # 6
    'traffsign',        # 7
    'vegetation',       # 8
    'terrain',          # 9
    'sky',              # 10
    'person',           # 11
    'rider',            # 12
    'car',              # 13
    'truck',            # 14
    'bus',              # 15
    'train',            # 16
    'motorcycle',       # 17
    'bicycle',          # 18
    'unlabeled'         # 19
]
synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class Eval:
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
        self.ignore_index = None
        self.synthia = True if num_class == 16 else False

    def Pixel_Accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return PA

    def Mean_Pixel_Accuracy(self, out_16_13=False):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.synthia:
            MPA_16 = np.nanmean(MPA[:self.ignore_index])
            MPA_13 = np.nanmean(MPA[synthia_set_16_to_13])
            return MPA_16, MPA_13
        if out_16_13:
            MPA_16 = np.nanmean(MPA[synthia_set_16])
            MPA_13 = np.nanmean(MPA[synthia_set_13])
            return MPA_16, MPA_13
        MPA = np.nanmean(MPA[:self.ignore_index])

        return MPA

    def Mean_Intersection_over_Union(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        if self.synthia:
            MIoU_16 = np.nanmean(MIoU[:self.ignore_index])
            MIoU_13 = np.nanmean(MIoU[synthia_set_16_to_13])
            return MIoU_16, MIoU_13
        if out_16_13:
            MIoU_16 = np.nanmean(MIoU[synthia_set_16])
            MIoU_13 = np.nanmean(MIoU[synthia_set_13])
            return MIoU_16, MIoU_13
        MIoU = np.nanmean(MIoU[:self.ignore_index])

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self, out_16_13=False):
        FWIoU = np.multiply(np.sum(self.confusion_matrix, axis=1), np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                         np.diag(self.confusion_matrix))
        if self.synthia:
            FWIoU_16 = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_16_to_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        if out_16_13:
            FWIoU_16 = np.sum(i for i in FWIoU[synthia_set_16] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)

        return FWIoU

    def Mean_Precision(self, out_16_13=False):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        if self.synthia:
            Precision_16 = np.nanmean(Precision[:self.ignore_index])
            Precision_13 = np.nanmean(Precision[synthia_set_16_to_13])
            return Precision_16, Precision_13
        if out_16_13:
            Precision_16 = np.nanmean(Precision[synthia_set_16])
            Precision_13 = np.nanmean(Precision[synthia_set_13])
            return Precision_16, Precision_13
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision

    def Print_Every_class_Eval(self, out_16_13=False, logger=None):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Class_ratio = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        if logger is not None:
            logger.info('===>Everyclass:\t' + 'MPA\t' + 'MIoU\t' + 'PC\t' + 'Ratio\t' + 'Pred_Retio')
        else:
            print('===>Everyclass:\t' + 'MPA\t' + 'MIoU\t' + 'PC\t' + 'Ratio\t' + 'Pred_Retio')
        if out_16_13: MIoU = MIoU[synthia_set_16]
        ##
        name_classes_eval = [name_classes[i] for i in synthia_set_16] if self.synthia else name_classes
        class_name_str = ''
        MIoU_str = ''
        ##
        for ind_class in range(len(MIoU)):
            pa = str(round(MPA[ind_class] * 100, 2)) if not np.isnan(MPA[ind_class]) else 'nan'
            iou = str(round(MIoU[ind_class] * 100, 2)) if not np.isnan(MIoU[ind_class]) else 'nan'
            pc = str(round(Precision[ind_class] * 100, 2)) if not np.isnan(Precision[ind_class]) else 'nan'
            cr = str(round(Class_ratio[ind_class] * 100, 2)) if not np.isnan(Class_ratio[ind_class]) else 'nan'
            pr = str(round(Pred_retio[ind_class] * 100, 2)) if not np.isnan(Pred_retio[ind_class]) else 'nan'
            class_name_str += name_classes_eval[ind_class] + '\t'
            MIoU_str += iou + '\t'
            if logger is not None:
                logger.info('===>' + name_classes_eval[ind_class] + ':\t' + pa + '\t' + iou + '\t' + pc + '\t' + cr + '\t' + pr)
            else:
                print('===>' + name_classes_eval[ind_class] + ':\t' + pa + '\t' + iou + '\t' + pc + '\t' + cr + '\t' + pr)
        logger.info(class_name_str)
        logger.info(MIoU_str)

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape

        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def computer_and_save_metric(name, eval, logger, writer, current_epoch, num_classes):
    # computer metric
    if num_classes == 16:
        PA = eval.Pixel_Accuracy()
        MPA_16, MPA_13 = eval.Mean_Pixel_Accuracy()
        MIoU_16, MIoU_13 = eval.Mean_Intersection_over_Union()
        FWIoU_16, FWIoU_13 = eval.Frequency_Weighted_Intersection_over_Union()
        PC_16, PC_13 = eval.Mean_Precision()
        logger.info('Results of {}, PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(name, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
        logger.info('Results of {}, PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(name, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
        writer.add_scalar(name + '/PA', PA, current_epoch)
        writer.add_scalar(name + '/MPA_16', MPA_16, current_epoch)
        writer.add_scalar(name + '/MIoU_16', MIoU_16, current_epoch)
        writer.add_scalar(name + '/FWIoU_16', FWIoU_16, current_epoch)
        writer.add_scalar(name + '/MPA_13', MPA_13, current_epoch)
        writer.add_scalar(name + '/MIoU_13', MIoU_13, current_epoch)
        writer.add_scalar(name + '/FWIoU_13', FWIoU_13, current_epoch)
        return PA, MPA_16, MIoU_16, FWIoU_16
    else:
        PA = eval.Pixel_Accuracy()
        MPA = eval.Mean_Pixel_Accuracy()
        MIoU = eval.Mean_Intersection_over_Union()
        FWIoU = eval.Frequency_Weighted_Intersection_over_Union()
        PC = eval.Mean_Precision()
        logger.info('Results of {}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(name, PA, MPA, MIoU, FWIoU, PC))
        writer.add_scalar(name + '/PA', PA, current_epoch)
        writer.add_scalar(name + '/MPA', MPA, current_epoch)
        writer.add_scalar(name + '/MIoU', MIoU, current_epoch)
        writer.add_scalar(name + '/FWIoU', FWIoU, current_epoch)
        return PA, MPA, MIoU, FWIoU
