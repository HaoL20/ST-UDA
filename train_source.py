import os

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu', type=str, default="0", help=" the num of gpu")
arg_parser.add_argument('--conf', type=str, default="config/config_eval.yaml", help=" the num of gpu")
opt = arg_parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

import yaml
import shutil
import random
import logging
import torch.nn
import torch.nn.functional
import torch.backends.cudnn
from tqdm import tqdm
from math import ceil
from tensorboardX import SummaryWriter

from misc.eval import Eval
from misc.utils import *
from misc.eval import computer_and_save_metric


class Trainer:
    def __init__(self, conf, device, logger):
        self.conf = conf
        self.device = device
        self.logger = logger
        self.model_name = conf['source_data_name']  # 保存的模型前缀

        self.ignore_index = -1
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.best_iter = None

        # set TensorboardX
        self.writer = SummaryWriter(self.conf['checkpoint_dir'])

        # Metric definition
        self.eval = Eval(self.conf['num_classes'])

        # loss definition
        self.loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=-1)
        self.loss.to(self.device)

        # model
        self.model, params = get_model(conf)
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        if conf['optim'] == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=self.conf['lr'], momentum=conf['momentum'], weight_decay=conf['weight_decay'])
        elif conf['optim'] == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=self.conf['lr'], betas=(conf['beta'], 0.99), weight_decay=conf['weight_decay'])

        # data loader
        self.source_train_loader, self.source_val_loader = get_dataloader(conf, 'source')
        self.target_train_loader, self.target_val_loader = get_dataloader(conf, 'target')

        self.train_iterations = (len(self.source_train_loader.dataset) + conf['batch_size']) // conf['batch_size']  # 一个epoch的迭代次数
        self.train_iterations = min(conf['iter_epoch'], self.train_iterations)  # 一个epoch的迭代次数最多不能超过iter_epoch

        if conf['iter_stop'] is None:  # 总epoch数量
            self.num_epoch = ceil(conf['iter_total'] / self.train_iterations)
        else:
            self.num_epoch = ceil(conf['iter_stop'] / self.train_iterations)

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

        # load pretrained checkpoint
        if self.conf['warmup_model_file'] != '':
            if os.path.isdir(self.conf['warmup_model_file']):
                self.conf['warmup_model_file'] = os.path.join(self.conf['checkpoint_dir'], self.model_name + 'best.pth')
            self.load_checkpoint(self.conf['warmup_model_file'])

        self.train()
        self.writer.close()

    def train(self):

        for _ in tqdm(range(self.current_epoch, self.num_epoch), desc="Total {} epochs".format(self.num_epoch)):
            self.current_epoch += 1

            # train
            self.train_one_epoch()  # 训练一个epoch

            # validate
            PA, MPA, MIoU, FWIoU = self.validate('target_val')  # 使用target域进行评估，域自适应实验只关心在 target 域的结果
            # PA, MPA, MIoU, FWIoU= self.validate('source_val')     # 使用source域进行评估

            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.current_iter
                self.logger.info("=>saving a new best checkpoint...")
                self.save_checkpoint(self.model_name + 'best.pth')
            else:
                self.logger.info("=> The MIoU of val doesn't improve.")
                self.logger.info("=> The best MIoU of val is {} at {}".format(self.best_MIou, self.best_iter))
            self.logger.info('========================= epoch-{} end ========================='.format(self.current_epoch))

        self.logger.info("=> best_MIou {} at {}".format(self.best_MIou, self.best_iter))
        self.logger.info("=> Saving the final checkpoint to " + os.path.join(self.conf['checkpoint_dir'], self.model_name + 'final.pth'))
        self.save_checkpoint(self.model_name + 'final.pth')

    def train_one_epoch(self):
        tqdm_epoch = tqdm(self.source_train_loader, total=self.train_iterations, desc="Train epoch-{}-total-{}".format(self.current_epoch, self.num_epoch))
        self.logger.info("")
        self.logger.info("Training on source_train of epoch-{}...".format(self.current_epoch))
        self.eval.reset()  # 混淆矩阵清零
        train_loss = []

        self.model.train()

        for batch_idx, data in enumerate(tqdm_epoch):

            # 数据处理
            images, labels, id_labels = data
            images, labels = images.to(self.device), labels.to(device=self.device, dtype=torch.long)  # 尺寸大小：(b,3,h,w), (b,h,w)

            # 更新学习率
            poly_lr_scheduler(self.optimizer, self.conf['lr'], self.current_iter, self.conf['iter_total'], self.conf['poly_power'])
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            self.optimizer.zero_grad()                  # 梯度清零
            preds, _ = self.model(images)               # 前向传播，返回prediction, feature
            cur_loss = self.loss(preds, labels)         # 交叉熵损失函数
            cur_loss.backward()                         # 反向传播
            self.optimizer.step()                       # 参数更新

            # 损失可视化
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            self.writer.add_scalar('train_loss', cur_loss, self.current_iter)

            # 更新混淆矩阵
            arg_preds = torch.argmax(preds, dim=1)                      # 预测结果，(b,c,h,w) ==> (b,h,w)
            self.eval.add_batch(labels.cpu().numpy(), arg_preds.cpu().numpy())

            # 计算当前 epoch 的平均 loss
            train_loss.append(cur_loss.item())
            train_loss_avg = sum(train_loss) / len(train_loss)

            self.current_iter += 1  # 更新总迭代次数

            # 打印当前 loss
            if batch_idx % self.conf['iter_show_loss'] == 0:
                self.logger.info("The train loss of epoch-{}-batch-{}:{}".format(self.current_epoch, batch_idx, cur_loss.item()))

            # epoch 迭代结束
            if batch_idx == self.train_iterations - 1:
                self.save_images(images, labels, arg_preds, 'source_train')  # 可视化最后一次迭代的图片
                tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, train_loss_avg))  # 打印当前 epoch 的平均 loss
                tqdm_epoch.close()
                break

        # 计算指标
        computer_and_save_metric('source_train', self.eval, self.logger, self.writer, self.current_epoch, self.conf['num_classes'])

    def validate(self, name='target_val'):
        self.logger.info('Validating on {} of epoch-{}....'.format(name, self.current_epoch))
        self.eval.reset()

        val_loader = self.target_val_loader if name == 'target_val' else self.source_val_loader

        with torch.no_grad():
            tqdm_batch = tqdm(val_loader, desc="Val epoch-{}-".format(self.current_epoch))
            self.model.eval()

            for batch_idx, data in enumerate(tqdm_batch):
                # unpack data
                images, labels, id_labels = data

                b, H, W = labels.shape
                images, labels = images.to(self.device), labels.to(device=self.device, dtype=torch.long)    # (b,3,h,w), (b,H,W) val评估的时候，需要原图大小的标签

                preds, _ = self.model(images)                                                               # prediction, feature
                arg_preds = torch.argmax(preds, dim=1, keepdim=True)                                        # 预测类别, (b,c,h,w) ==> (b,1,h,w) (要对h，w上采样，interpolate输入必须为4维。因此要keepdim)
                arg_preds = arg_preds.to(torch.float32)                                                     # interpolate 不能处理int类型
                arg_preds = torch.nn.functional.interpolate(arg_preds, size=(H, W), mode='nearest')         # (b,1,h,w) ==> (b,1,H,W) 上采样
                arg_preds = arg_preds.squeeze(dim=1).to(torch.int64)                                        # (b,1,H,W) ==> (b,H,W), 和labels数据类型一致

                self.eval.add_batch(labels.cpu().numpy(), arg_preds.cpu().numpy())

            #  可视化最后一次迭代的图片
            self.save_images(images, labels, arg_preds, name)
            PA, MPA, MIoU, FWIoU = computer_and_save_metric(name, self.eval, self.logger, self.writer, self.current_epoch, self.conf['num_classes'])
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def save_images(self, images, labels, arg_preds, name):

        # show train image on tensorboard
        images_inv = inv_preprocess(images.clone().cpu())
        labels_colors = decode_labels(labels, self.conf['num_classes'])
        preds_colors = decode_labels(arg_preds, self.conf['num_classes'])
        self.writer.add_image('{}/Images'.format(name), images_inv, self.current_epoch)
        self.writer.add_image('{}/Labels'.format(name), labels_colors, self.current_epoch)
        self.writer.add_image('{}/preds'.format(name), preds_colors, self.current_epoch)

    def save_checkpoint(self, file_name=None):
        file_name = os.path.join(self.conf['checkpoint_dir'], file_name)
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }
        torch.save(state, file_name)

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from " + filename)
        except OSError:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.conf['checkpoint_dir']))
            self.logger.info("**First time to train**")


def init_config(config_path):
    # Reading configuration file
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    train_conf = config['train']

    # data configure
    for domain_data_name in ['source_data_name', 'target_data_name']:
        data_name = train_conf[domain_data_name]  # 数据集名称
        data_conf = train_conf[data_name]  # 数据集参数
        data_conf['size'] = tuple(map(int, data_conf['size'].split(',')))  # 将size的h,w转换为int，并且组合为tuple类型

    assert train_conf['num_classes'] == 19 if train_conf['source_data_name'] == 'gta5' else train_conf['num_classes'] == 16

    # checkpoint_dir configure
    checkpoint_dir = train_conf['checkpoint_dir']
    if os.path.exists(checkpoint_dir):
        key_str = input("{}已经存在！\n删除该文件夹请输入：d\n忽略请输入：c\n结束请输入：e\n请选择:".format(checkpoint_dir))
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
    fh = logging.FileHandler(os.path.join(checkpoint_dir, 'train_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 保证可复现性

    random.seed(train_conf['seed'])
    np.random.seed(train_conf['seed'])

    torch.random.manual_seed(train_conf['seed'])
    torch.manual_seed(train_conf['seed'])  # 为CPU设置随机种子
    torch.cuda.manual_seed(train_conf['seed'])  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(train_conf['seed'])  # 为所有GPU设置随机种子

    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(train_conf['seed'])

    # torch.backends.cudnn.deterministic = True
    # # torch.backends.cudnn.benchmark = False
    # # 可复现设计细节参考：
    # # https://zhuanlan.zhihu.com/p/73711222
    # # https://blog.csdn.net/weixin_42587961/article/details/109363698
    # # https://www.jianshu.com/p/1b9e18146045
    # # https://www.it610.com/article/1293854112777052160.htm
    return train_conf, device, logger


def main():
    # config_path = "config/config_train_source.yaml"
    config_path = opt.conf
    train_conf, logger, device = init_config(config_path)
    agent = Trainer(train_conf, logger, device)
    agent.main()


if __name__ == '__main__':
    main()
