import os
import yaml
import shutil
import logging
import torch.nn
import torch.nn.functional
import torch.backends.cudnn
from tqdm import tqdm
from tensorboardX import SummaryWriter

from misc.eval import Eval
from misc.utils import *
from misc.eval import computer_and_save_metric

class Evaluater():
    def __init__(self, conf, device, logger):
        self.conf = conf
        self.device = device
        self.logger = logger

        self.checkpoint_dir = conf['checkpoint_dir']
        self.id_mask_dir = os.path.dirname(os.path.join(self.checkpoint_dir, 'id_mask'))
        self.color_mask_dir = os.path.dirname(os.path.join(self.checkpoint_dir, 'color_mask'))
        if not os.path.exists(self.id_mask_dir):
            os.makedirs(self.id_mask_dir)
        if not os.path.exists(self.color_mask_dir):
            os.makedirs(self.color_mask_dir)

        # set TensorboardX
        self.writer = SummaryWriter(self.conf['checkpoint_dir'])

        # Metric definition
        self.eval = Eval(conf['num_classes'])

        # model
        self.model, _ = get_model(conf)
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.load_checkpoint(conf['model_file'])

        # dataloader
        _, self.val_loader = get_dataloader(conf, domain='target')

    def main(self):

        # display config details
        self.logger.info("Global configuration as follows:")
        for key, val in self.conf.items():
            self.logger.info("{:16} {}".format(key, val))

        # choose cuda
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        # validate
        self.validate()

        self.writer.close()

    def validate(self):
        self.logger.info('validating one epoch...')
        self.eval.reset()

        with torch.no_grad():
            tqdm_batch = tqdm(self.val_loader, desc="Val Epoch")
            self.model.eval()

            for batch_idx, data in enumerate(tqdm_batch):
                # unpack data
                images, labels, id_labels = data


                images, labels = images.to(self.device), labels.to(device=self.device, dtype=torch.long)    # (b,3,h,w), (b,H,W) val评估的时候，需要原图大小的标签
                labels = torch.squeeze(labels, 1)                                                           # (b,H,W) ==> (b,1,H,W)

                preds, _ = self.model(images)  # prediction, feature
                probs = torch.nn.functional.softmax(preds, dim=1)

                if self.conf['flip']:
                    preds_flip, _ = self.model(torch.flip(images, dims=[3]))     # 水平翻转
                    probs_flip = torch.nn.functional.softmax(preds_flip, dim=1)
                    probs = (probs+probs_flip)/2

                arg_probs = torch.argmax(probs, dim=1, keepdim=True)                                        # 预测类别, (b,c,h,w) ==> (b,1,h,w) (要对h，w上采样，interpolate输入必须为4维。因此要keepdim)
                arg_probs = arg_probs.to(torch.float32)                                                     # interpolate 不能处理int类型
                if self.conf['evl_with_original']:
                    b, H, W = labels.shape
                    arg_probs = torch.nn.functional.interpolate(arg_probs, size=(H, W), mode='nearest')     # (b,1,h,w) ==> (b,1,H,W) 上采样
                else:
                    b, _, h, w = arg_probs.shape
                    labels = torch.nn.functional.interpolate(labels.unsqueeze(dim=1), size=(h, w), mode='nearest')           # (b,H,W) ==> (b,1,H,W) ==> (b,1,h,w) 上采样
                arg_probs = arg_probs.squeeze(dim=1).to(torch.long)                                         # (b,1,H,W) ==> (b,H,W), 和labels数据类型一致
                labels = labels.squeeze(dim=1).to(torch.long)                                               # (b,1,H,W) ==> (b,H,W)
                self.eval.add_batch(labels.cpu().numpy(), arg_probs.cpu().numpy())

                for i in range(b):
                    save_prob = arg_probs[i].cpu().data.numpy()
                    save_prob = np.asarray(save_prob, dtype=np.uint8)
                    output_col = colorize_mask(save_prob, self.conf['num_classes'])
                    output = Image.fromarray(save_prob)

                    label_name = id_labels[0].split('/')[-1]

                    output.save(os.path.join(self.checkpoint_dir, label_name))
                    output_col.save(os.path.join(self.checkpoint_dir, label_name))

                # if batch_idx % 20 == 0:
                #     self.save_images(images, labels, arg_probs, 'target_val')  # 可视化最后一次迭代的图片

            PA, MPA, MIoU, FWIoU = computer_and_save_metric('target_val', self.eval, self.logger, self.writer, self.conf['num_classes'])
            self.eval.Print_Every_class_Eval(logger = self.logger)
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    # def save_images(self, images, labels, arg_preds, name):
    #
    #     # show train image on tensorboard
    #     images_inv = inv_preprocess(images.clone().cpu())
    #     labels_colors = decode_labels(labels, self.conf['num_classes'])
    #     preds_colors = decode_labels(arg_preds, self.conf['num_classes'])
    #     self.writer.add_image('{}/Images'.format(name), images_inv, self.current_epoch)
    #     self.writer.add_image('{}/Labels'.format(name), labels_colors, self.current_epoch)
    #     self.writer.add_image('{}/preds'.format(name), preds_colors, self.current_epoch)

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=torch.device(self.device))

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)
        except OSError as _:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filename))


def init_config(config_path):
    # Reading configuration file
    config = yaml.load(open(config_path, "r", encoding='utf-8'), Loader=yaml.FullLoader)
    train_conf = config['train']

    # data configure
    data_name = train_conf['data_name']  # 数据集名称
    data_conf = train_conf[data_name]  # 数据集参数
    data_conf['size'] = tuple(map(int, data_conf['size'].split(',')))  # 将size的h,w转换为int，并且组合为tuple类型

    assert train_conf['num_classes'] == 19 if data_name in ['gta5', 'cityscapes'] else train_conf['num_classes'] == 16

    # checkpoint_dir configure
    checkpoint_dir = train_conf['checkpoint_dir']
    if os.path.exists(checkpoint_dir):
        key_str = input("删除该文件夹请输入：d\n忽略请输入：c\n结束请输入：e\n请选择:")
        if key_str == 'd':
            shutil.rmtree(checkpoint_dir)
            print("remove {} successfully".format(checkpoint_dir))
        elif key_str == 'c':
            print("continue training！")
        elif key_str == 'e':
            print('exit!')
            exit(0)
        else:
            print('error input')
            exit(0)
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except FileNotFoundError:
            print('Missing parent folder in path:  {}'.format(checkpoint_dir))
            exit()

    # save config
    shutil.copy(config_path, checkpoint_dir)

    # Device configure
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(checkpoint_dir, 'eval_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 保证可复现性
    torch.backends.cudnn.benchmark = True

    return train_conf, device, logger


def main():
    config_path = "config/config_eval.yaml"
    train_conf, logger, device = init_config(config_path)
    agent = Evaluater(train_conf, logger, device)
    agent.main()


if __name__ == '__main__':
    main()