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
        self.num_classes = num_classes          # 类别数，GTA5-to-Cityscapes为19
        self.ignore_label = ignore_label        # 忽略的标签序号

        # 图像增强
        self.color_jitter = color_jitter        # 色彩抖动
        self.random_mirror = random_mirror      # 随机镜像
        self.gaussian_blur = gaussian_blur      # 高斯模糊

        self.is_train = (self.split == 'train')
        if self.is_train and self.resize is False:
            self.label_dir = label_dir['resize']      # 如果数据集已经resize过，并且是训练阶段的话，用resize后的标签路径
        else:
            self.label_dir = label_dir['no_resize']     # 如果是测试阶段，或者是没有resize过的话，用no_resize的标签路径

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
        image, label = self._sync_transform(image=image, label=label, is_train=self.is_train)

        return image, label, id_label

    def __len__(self):
        return len(self.items)

    def _sync_transform(self, image, label, is_train, mirror_p=0.5, color_jitter_p=0.5, gaussian_p=0.5):

        image_transforms_list = [transforms.ToTensor(),                                         # (h,w,c) ==> (c,h,w) Converts [0,255] to [0,1]
                                 transforms.Normalize([.485, .456, .406], [.229, .224, .225])]  # ImageNet数据集的均值和方差。因为使用了ImageNet预训练模型，数据应该进行相同的Normalize

        if self.resize:
            image = image.resize(self.size, Image.BICUBIC)
            if self.is_train:
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


def test_dataset():
    os.chdir('..')  # 改变当前工作目录到上一级目录(项目目录)

    # Reading configuration file
    config = yaml.load(open("config/config_train_source.yaml", "r"), Loader=yaml.FullLoader)
    train_conf = config['train']
    conf_syn = train_conf['synthia']

    conf_syn['size'] = tuple(map(int, conf_syn['size'].split(',')))  # 将size的h,w转换为int，并且组合为tuple类型
    if conf_syn['use_trans_image'] is True:
        conf_syn['image_dir'] = conf_syn['trans_image_dir']

    train_loader = get_syn_dataloader(config['train'], split='val')
    if os.path.exists('datasets/demo_img/syn/') is False:
        os.makedirs('datasets/demo_img/syn/')

    for idx, pack_data in enumerate(train_loader):
        images, labels, id_labels = pack_data  # (b,3,h,w), (b,h,w)
        # 文件名
        print(id_labels)

        # 可视化图像
        img = torchvision.utils.make_grid(images, nrow=4, normalize=True).numpy()  # 使用make_grid的normalize
        img = np.transpose(img, (1, 2, 0)) * 255
        img = Image.fromarray(np.uint8(img))
        img.save('datasets/demo_img/syn/syn_Demo_{}.jpg'.format(idx))

        # 可视化标签
        labels = torch.unsqueeze(labels, dim=1)  # (b,h,w)   ==> (b,1,h,w)，make_grid只能处理4维的向量，三维的label必须扩充一个通道的维度
        labels = torchvision.utils.make_grid(labels, nrow=4)  # (b,1,h,w) ==> (3,h,w), 单通道会被扩充到3通道。
        labels = labels.numpy()[0]  # (3,h,w)   ==> (h,w)，  转换为numpy，取单通道。P模式的图片必须要单通道。

        output_col = utils.colorize_mask(labels, train_conf['num_classes'])  # 转换为P模式的Image，并且换上对应的调试板，将其可视化。
        output_col.save('datasets/demo_img/syn/syn_Demo_label_{}.png'.format(idx))

        if idx > 2:
            break


if __name__ == '__main__':
    test_dataset()
