import torch
import numpy as np
from PIL import Image
from models.DeepLabV2_feat import DeeplabFeat

# class 19 colour map
label_colours_19 = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]]

# class 16 colour map
class19_to_class16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19]    #
label_colours_16 = [label_colours_19[idx] for idx in class19_to_class16]


def inv_preprocess(image):
    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    norm_ip(image, float(image.min()), float(image.max()))
    return image


def colorize_mask(mask, num_classes):       # mask的尺寸为（H，W），没有其他维度
    assert num_classes == 16 or num_classes == 19
    label_colours = label_colours_16 if num_classes == 16 else label_colours_19
    palettes = []
    for label_colour in label_colours:
        palettes = palettes + list(label_colour)
    palettes = palettes + [255, 255, 255] * (256 - len(palettes))

    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
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
