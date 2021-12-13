# -*- coding: utf-8 -*-
import os
import sys
import yaml
import torch
import random
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageFilter, ImageFile

sys.path.append("..")
from utils import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


class City_Dataset(data.Dataset):
    def __init__(self, image_dir, label_dir, size, num_class, ignore_label=-1, split='train', resize=True,
                 use_pseudo=False, gaussian_blur=False, color_jitter=False, random_mirror=False):

        self.size = size                        # resize大小
        self.split = split                      # train,val
        self.resize = resize                    # 是否resize
        self.image_dir = image_dir              # 数据集图像路径，eg: home/haol/data/Dataset/cityscape/gtFine
        self.label_dir = label_dir              # 数据集标签路径，eg：/home/haol/data/Dataset/cityscape/leftImg8bit
        self.num_class = num_class              # 类别数，GTA5-to-Cityscapes为19，SYNTHIA-to-Cityscapes为16
        self.use_pseudo = use_pseudo            # 是否加载伪标签（加载伪标签的情况下，读取的伪标签已经映射好了，只需要将255映射为ignore_label）
        self.ignore_label = ignore_label        # 忽略的标签序号

        # 图像增强
        self.color_jitter = color_jitter        # 色彩抖动
        self.random_mirror = random_mirror      # 随机镜像
        self.gaussian_blur = gaussian_blur      # 高斯模糊

        self.gen_pseudo = False                 # 是否用于生成伪标签，默认为False,通过switch_to_gen_pseudo切换为True

        # GTA5-to-Cityscapes 实验中，只考虑共享的19类
        self.id_to_train_id = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                               19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                               26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # SYNTHIA-to-Cityscapes 实验中，只考虑共享的16类
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]                # 19类中共享的16类
        self.train_id_to_16id = {id_19: id_16 for id_16, id_19 in enumerate(synthia_set_16)}    # 19类到16类的映射关系
        self.train_id_to_16id[255] = ignore_label

        # 获取数据集路径列表
        self.items = [i for i in open(os.path.join('Cityscapes', self.split + ".txt"))]

    def __getitem__(self, index):
        id_image, id_label = self.items[index].strip('\n').split(' ')
        image_path = self.image_dir + id_image
        label_path = self.label_dir + id_label
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        image, label = self._sync_transform(image=image, label=label, is_train=(self.split == 'train'))

        return image, label, id_label

    def __len__(self):
        return len(self.items)

    def _sync_transform(self, image, label, is_train, mirror_p=0.5, color_jitter_p=0.5, gaussian_p=0.5):

        image_transforms_list = [transforms.ToTensor(),                                         # (h,w,c) ==> (c,h,w) Converts [0,255] to [0,1]
                                 transforms.Normalize([.485, .456, .406], [.229, .224, .225])]  # ImageNet数据集的均值和方差。因为使用了ImageNet预训练模型，数据应该进行相同的Normalize

        if self.resize:
            image = image.resize(self.size, Image.BICUBIC)
            label = label.resize(self.size, Image.NEAREST)

        # 随机翻转
        if self.random_mirror and random.random() < mirror_p and is_train:
            print(self.random_mirror, self.gen_pseudo)
            assert not (self.gen_pseudo == self.random_mirror and self.gen_pseudo is True)  # 生成伪标签的时候，不能random_mirror
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 高斯模糊
        if self.gaussian_blur and random.random() < gaussian_p and is_train:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 1.15)))

        # 色彩抖动
        if self.color_jitter and random.random() < color_jitter_p and is_train:
            image_transforms_list.insert(0, transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))

        # image转tensor
        image_transforms = transforms.Compose(image_transforms_list)
        image = image_transforms(image)

        # label映射、转tensor
        label = np.asarray(label, np.float32)
        label = self._label_mapping(label)
        label = torch.from_numpy(label.copy())

        return image, label

    def _label_mapping(self, label):
        if self.use_pseudo:                         # 伪标签已经做好了映射，只需要将255索引改成ignore_label
            label[label == 255] = self.ignore_label
            return label

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_train_id.items():    # 转换为train_id
            label_copy[label == k] = v
        if self.num_class == 16:                    # 转换为16类
            label_copy_16 = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.train_id_to_16id.items():
                label_copy_16[label_copy == k] = v
            label_copy = label_copy_16
        return label_copy

    def switch_to_gen_pseudo(self):     # 切换到生成伪标签，不能在迭代过程中切换
        assert self.split == 'train'    # 只能在train模式下生成伪标签
        self.gen_pseudo = True
        self.random_mirror = False      # 生成伪标签的时候不使用random_mirror
        print("Successfully switch to generate pseudo stata")


def get_city_dataloader(conf, split):
    assert split in ['train', 'val']
    conf_city = conf['city']
    data_set = City_Dataset(split=split,
                            size=conf_city['size'],
                            resize=conf_city['resize'],
                            image_dir=conf_city['image_dir'],
                            label_dir=conf_city['label_dir'],
                            num_class=conf['num_class'],
                            use_pseudo=conf['use_pseudo'],
                            ignore_label=conf['ignore_label'],
                            color_jitter=conf['color_jitter'],
                            gaussian_blur=conf['gaussian_blur'],
                            random_mirror=conf['random_mirror'])

    data_loader = data.DataLoader(dataset=data_set,
                                  batch_size=conf['batch_size'],
                                  pin_memory=conf['pin_memory'],
                                  num_workers=conf['num_workers'],
                                  shuffle=(split == 'train'),
                                  drop_last=True)

    return data_loader


class City_DataLoader:
    def __init__(self, conf):
        conf_city = conf['city']
        # 训练集
        data_set = City_Dataset(split='train',
                                size=conf_city['size'],
                                resize=conf_city['resize'],
                                image_dir=conf_city['image_dir'],
                                label_dir=conf_city['label_dir'],
                                num_class=conf['num_class'],
                                use_pseudo=conf['use_pseudo'],
                                ignore_label=conf['ignore_label'],
                                color_jitter=conf['color_jitter'],
                                gaussian_blur=conf['gaussian_blur'],
                                random_mirror=conf['random_mirror'])

        self.train_loader = data.DataLoader(dataset=data_set,
                                            batch_size=conf['batch_size'],
                                            pin_memory=conf['pin_memory'],
                                            num_workers=conf['num_workers'],
                                            shuffle=True,
                                            drop_last=True)

        # 评估集
        val_set = City_Dataset(split='val',
                               size=conf_city['size'],
                               resize=conf_city['resize'],
                               image_dir=conf_city['image_dir'],
                               label_dir=conf_city['label_dir'],
                               num_class=conf['num_class'],
                               use_pseudo=conf['use_pseudo'],
                               ignore_label=conf['ignore_label'],
                               color_jitter=conf['color_jitter'],
                               gaussian_blur=conf['gaussian_blur'],
                               random_mirror=conf['random_mirror'])

        self.val_loader = data.DataLoader(dataset=val_set,
                                          batch_size=conf['batch_size'],
                                          pin_memory=conf['pin_memory'],  # 当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。
                                          num_workers=conf['num_workers'],
                                          shuffle=False,
                                          drop_last=True)

        self.train_iterations = len(val_set) // conf['batch_size'] + 1  # 训练集一个epoch迭代次数
        self.valid_iterations = len(val_set) // conf['batch_size'] + 1  # 评估集一个epoch迭代次数


if __name__ == '__main__':
    # Reading configuration file
    config = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)
    train_conf = config['train']
    train_conf['city']['size'] = tuple(map(int, train_conf['city']['size'].split(',')))     # 将size的h,w转换为int，并且组合为tuple类型
    train_loader = get_city_dataloader(config['train'], split='val')

    if os.path.exists('demo_img/city/') is False:
        os.makedirs('demo_img/city/')

    for idx, data in enumerate(train_loader):
        images, labels, id_labels = data  # (b,3,h,w), (b,h,w)
        # 文件名
        print(id_labels)

        # 可视化图像
        # 方法1
        # Normalize的图像复原，归一化到之前的【0，1】之间
        mean = torch.as_tensor([.485, .456, .406], dtype=images.dtype, device=images.device).view(-1, 1, 1)  # (3) ==> (3,1,1)，这样才可以进行后面的广播运算
        std = torch.as_tensor([.229, .224, .225], dtype=images.dtype, device=images.device).view(-1, 1, 1)
        img = images * std + mean

        img = torchvision.utils.make_grid(img, nrow=4).numpy()                      # (b,3,h,w) ==> (3,H,W)。其中H，W是把b张图片按照相应规则(每行最多4张图片)拼接成的新图片的尺寸。
        img = np.transpose(img, (1, 2, 0)) * 255                                    # (3,H,W) ==> (H,W,3)
        # img = img[:, :, ::-1]                                                     # 如果加载为BGR的模式，需要转换为RGB的模式
        img = Image.fromarray(np.uint8(img))                                        # 转换为uint8，再转换为Image
        img.save('demo_img/city/Cityscape_Demo_{}.jpg'.format(idx))

        # 方法2
        # make_grid函数也支持normalize复原，normalize=True即可。
        # 但是是通过(img-min)/(max - min + 1e-5)将img归一化到【0，1】，会和用mean、std归一化的图像有些许差别
        img = torchvision.utils.make_grid(images, nrow=4, normalize=True).numpy()   # 使用make_grid的normalize
        img = np.transpose(img, (1, 2, 0)) * 255
        # img = img[:, :, ::-1]
        img = Image.fromarray(np.uint8(img))
        img.save('demo_img/city/Cityscape_Demo_{}_method_2.jpg'.format(idx))

        # 可视化标签
        labels = torch.unsqueeze(labels, dim=1)                                     # (b,h,w)   ==> (b,1,h,w)，make_grid只能处理4维的向量，三维的label必须扩充一个通道的维度
        labels = torchvision.utils.make_grid(labels, nrow=4)                        # (b,1,h,w) ==> (3,h,w), 单通道会被扩充到3通道。
        labels = labels.numpy()[0]                                                  # (3,h,w)   ==> (h,w)，  转换为numpy，取单通道。P模式的图片必须要单通道。
        output_col = utils.colorize_mask(labels, train_conf['num_class'])           # 转换为P模式的Image，并且换上对应的调试板，将其可视化。
        output_col.save('demo_img/city/Cityscape_Demo_label_{}.png'.format(idx))

        if idx > 2:
            break
