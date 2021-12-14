# -*- coding: utf-8 -*-
import os
import sys
import yaml
import torch
import random
import imageio
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageFilter, ImageFile

sys.path.append("..")
from utils import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SYNTHIA_Dataset(data.Dataset):
    def __init__(self, image_dir, label_dir, size, num_classes=16, ignore_label=-1, split='train',
                 resize=True, gaussian_blur=False, color_jitter=False, random_mirror=False):

        self.size = size                        # resize大小
        self.split = split                      # train,val
        self.resize = resize                    # 是否resize
        self.image_dir = image_dir              # 数据集图像路径，eg: /home/haol/data/Dataset/公开数据集/GTA5/images
        self.label_dir = label_dir              # 数据集标签路径，eg：/home/haol/data/Dataset/公开数据集/GTA5/labels
        self.num_classes = num_classes              # 类别数，GTA5-to-Cityscapes为19
        self.ignore_label = ignore_label        # 忽略的标签序号

        # 图像增强
        self.color_jitter = color_jitter        # 色彩抖动
        self.random_mirror = random_mirror      # 随机镜像
        self.gaussian_blur = gaussian_blur      # 高斯模糊

        # SYNTHIA-to-Cityscapes 实验中，只考虑共享的16类
        self.id_to_train_id = {1: 9, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8, 7: 5, 8: 12, 9: 7,
                               10: 10, 11: 15, 12: 14, 15: 6, 17: 11, 19: 13, 21: 3}

        # 获取数据集路径列表
        self.items = [i for i in open(os.path.join('datasets/SYNTHIA', self.split + ".txt"))]

    def __getitem__(self, index):
        id_image, id_label = self.items[index].strip('\n').split(' ')
        image_path = self.image_dir + id_image
        label_path = self.label_dir + id_label
        image = Image.open(image_path).convert("RGB")
        label = imageio.imread(label_path, format='PNG-FI')[:, :, 0]
        label = Image.fromarray(np.uint8(label))
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
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_train_id.items():  # 转换为train_id
            label_copy[label == k] = v
        return label_copy


def get_syn_dataloader(conf, split):
    assert split in ['train', 'val']
    conf_syn = conf['synthia']
    data_set = SYNTHIA_Dataset(split=split,
                               size=conf_syn['size'],
                               resize=conf_syn['resize'],
                               image_dir=conf_syn['image_dir'],
                               label_dir=conf_syn['label_dir'],
                               num_classes=conf['num_classes'],
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


if __name__ == '__main__':
    # Reading configuration file
    config = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)
    train_conf = config['train']
    conf_syn = train_conf['synthia']

    conf_syn['size'] = tuple(map(int, conf_syn['size'].split(',')))  # 将size的h,w转换为int，并且组合为tuple类型
    if conf_syn['use_trans_image'] is True:
        conf_syn['image_dir'] = conf_syn['trans_image_dir']

    train_loader = get_syn_dataloader(config['train'], split='train')
    if os.path.exists('demo_img/syn/') is False:
        os.makedirs('demo_img/syn/')

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

        img = torchvision.utils.make_grid(img, nrow=4).numpy()              # (b,3,h,w) ==> (3,H,W)。其中H，W是把b张图片按照相应规则(每行最多4张图片)拼接成的新图片的尺寸。
        img = np.transpose(img, (1, 2, 0)) * 255                            # (3,H,W) ==> (H,W,3)
        # img = img[:, :, ::-1]                                             # 如果加载为BGR的模式，需要转换为RGB的模式
        img = Image.fromarray(np.uint8(img))                                # 转换为uint8，再转换为Image
        img.save('demo_img/syn/syn_Demo_{}.jpg'.format(idx))

        # 方法2
        # make_grid函数也支持normalize复原，normalize=True即可。
        # 但是是通过(img-min)/(max - min + 1e-5)将img归一化到【0，1】，会和用mean、std归一化的图像有些许差别
        img = torchvision.utils.make_grid(images, nrow=4, normalize=True).numpy()   # 使用make_grid的normalize
        img = np.transpose(img, (1, 2, 0)) * 255
        # img = img[:, :, ::-1]
        img = Image.fromarray(np.uint8(img))
        img.save('demo_img/syn/syn_Demo_{}_method_2.jpg'.format(idx))

        # 可视化标签
        labels = torch.unsqueeze(labels, dim=1)                 # (b,h,w)   ==> (b,1,h,w)，make_grid只能处理4维的向量，三维的label必须扩充一个通道的维度
        labels = torchvision.utils.make_grid(labels, nrow=4)    # (b,1,h,w) ==> (3,h,w), 单通道会被扩充到3通道。
        labels = labels.numpy()[0]                              # (3,h,w)   ==> (h,w)，  转换为numpy，取单通道。P模式的图片必须要单通道。

        output_col = utils.colorize_mask(labels, train_conf['num_classes'])   # 转换为P模式的Image，并且换上对应的调试板，将其可视化。
        output_col.save('demo_img/syn/syn_Demo_label_{}.png'.format(idx))

        if idx > 2:
            break