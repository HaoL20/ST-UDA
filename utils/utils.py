import torch
import torchvision
import numpy as np
from PIL import Image
from models.DeepLabV2_feat import DeeplabFeat

from datasets.cityscapes_Dataset import get_city_dataloader
from datasets.gta5_Dataset import get_gta5_dataloader
from datasets.synthia_Dataset import get_syn_dataloader
# class 19 colour map
label_colours_19 = label_colours_19 = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (0, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)]

# class 16 colour map
class19_to_class16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19]    #
label_colours_16 = [label_colours_19[idx] for idx in class19_to_class16]


def inv_preprocess(images):
    mean = torch.as_tensor([.485, .456, .406], dtype=images.dtype, device=images.device).view(-1, 1, 1)  # (3) ==> (3,1,1)，这样才可以进行后面的广播运算
    std = torch.as_tensor([.229, .224, .225], dtype=images.dtype, device=images.device).view(-1, 1, 1)
    images = images * std + mean
    images = torchvision.utils.make_grid(images, nrow=4)
    return images


def decode_labels(mask, num_classes):                   # 将mask (b,h,w) 转换为彩色的mask(b,3,h,w)，并且make_grid拼接
    assert num_classes == 16 or num_classes == 19
    label_colours = label_colours_16 if num_classes == 16 else label_colours_19
    for label_colour in label_colours:
        tuple(label_colour)

    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    b, h, w = mask.shape
    outputs = np.zeros((b, h, w, 3), dtype=np.uint8)
    for i in range(b):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    color_labels = torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)  # (b, h, w, 3) ==> (b, 3, h, w)
    color_labels = torchvision.utils.make_grid(color_labels, nrow=4)
    return color_labels


def colorize_mask(mask, num_classes):       # mask的尺寸为（H，W），保存为P模式的Image

    # 设置调色板
    assert num_classes == 16 or num_classes == 19
    label_colours = label_colours_16 if num_classes == 16 else label_colours_19
    palettes = []
    for label_colour in label_colours:
        palettes = palettes + list(label_colour)
    palettes = palettes + [255, 255, 255] * (256 - len(palettes))

    # mask转图片
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')  # 转换为P模式
    new_mask.putpalette(palettes)
    return new_mask


def get_model(conf):
    if conf['backbone'] == "resnet101":
        model = DeeplabFeat(num_classes=conf['num_classes'], backbone='ResNet101', pretrained=conf['imagenet_pretrained'])
    elif conf['backbone'] == "vgg16":
        model = DeeplabFeat(num_classes=conf['num_classes'], backbone='VGG16', pretrained=conf['imagenet_pretrained'])
    else:
        raise ValueError('{} segmentation network is not allowed, choose from: resnet101 or vgg16'.format(conf['backbone']))

    params = model.optim_parameters(conf['lr'])
    return model, params


def get_dataloader(conf, domain='source'):
    if domain == 'target':
        target_train_loader = get_city_dataloader(conf, 'train')
        target_val_loader = get_city_dataloader(conf, 'val')
        return target_train_loader, target_val_loader
    else:
        source_train_loader, source_val_loader = None, None
        if conf['source_data_name'] == 'gta5':
            source_train_loader = get_gta5_dataloader(conf, 'train')
            source_val_loader = get_gta5_dataloader(conf, 'val')

        if conf['source_data_name'] == 'synthia':
            source_train_loader = get_syn_dataloader(conf, 'train')
            source_val_loader = get_syn_dataloader(conf, 'val')
        return source_train_loader, source_val_loader


def poly_lr_scheduler(optimizer, init_lr=None, iter=None,
                      max_iter=None, power=None):
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = 10 * new_lr