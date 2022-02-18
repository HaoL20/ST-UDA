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
from tensorboardX import SummaryWriter

from misc.eval import Eval
from misc.utils import *
from misc.losses import STLoss
from misc.eval import computer_and_save_metric


class Trainer:
    def __init__(self, conf, device, logger):
        self.conf = conf
        self.device = device
        self.logger = logger
        self.model_name = conf['source_data_name'] + '_2_' + conf['target_data_name']  # 保存的模型前缀

        self.ignore_index = -1
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.best_iter = None

        self.round_num = conf['round_num']                          # round 次数，每一轮新的 round 生成一次伪标签
        self.current_round = conf['init_round']                     # 当前 round
        self.epoch_each_round = conf['epoch_each_round']            # 每一轮的 epoch 数
        self.epoch_num = self.round_num * self.epoch_each_round     # 总 epoch 数
        self.use_uncertainty = conf['use_uncertainty']              # 是否使用不确定性估计
        self.lambda_entropy = conf['lambda_entropy']                # entropy系数
        self.lambda_stuff = conf['lambda_stuff']                    # stuff系数
        self.lambda_things = conf['lambda_things']                  # thing系数
        self.pseudo_label_dir = None                                # 当前round伪标签路径，设置为：checkpoint_dir/pseudo-label/current_round, 每一轮都重新生成伪标签，并且更新该路径

        feat_channel = 2048 if conf['backbone'] == 'resnet101' else 1024  # 中间特征图的通道数量，resnet101为2048，vgg16为1024

        self.feat_reg_ST_loss = STLoss(ignore_index=-1, num_class=conf['num_classes'],  #
                                       deque_capacity_factor=conf['deque_capacity_factor'],
                                       feat_channel=feat_channel, device=self.device)

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

        # 迭代次数定为  目标域  的一个epoch的迭代次数
        self.train_iterations = (len(self.target_train_loader.dataset) + conf['batch_size']) // conf['batch_size']  # 一个epoch的迭代次数
        self.train_iterations = min(conf['iter_epoch'], self.train_iterations)  # 一个epoch的迭代次数最多不能超过iter_epoch
        self.iter_total = self.train_iterations * self.epoch_num  # 总迭代次数, 不同于train_source. 这里的iter_total是自动计算出来的

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

        self.logger.info('Iter max: {} \nNumber of iterations: {}'.format(self.iter_total, self.train_iterations))

        self.train()
        self.writer.close()

    def train(self):
        for _ in range(self.current_round, self.round_num):
            self.logger.info("\n############## Begin {}/{} Round! #################\n".format(self.current_round + 1, self.round_num))
            self.logger.info("epoch_each_round: {}".format(self.epoch_each_round))
            self.pseudo_label_dir = os.path.join(self.conf['checkpoint_dir'], 'pseudo-label', str(self.current_round))  # 伪标签路径

            if not os.path.exists(self.pseudo_label_dir):  # 新建伪标签文件夹
                self.gen_pseudo_label()
            else:
                self.logger.info(self.pseudo_label_dir + ' exists! Skip gen pseudo label')  # 伪标签文件夹已存在，则跳过
            self.updata_target_dataloader()  # 更新目标域的dataloader
            self.train_one_round()
            self.current_round += 1

    def gen_pseudo_label(self):
        tqdm_epoch = tqdm(self.target_train_loader, total=self.train_iterations, desc="Generate pseudo label Epoch-{}-total-{}".format(self.current_epoch + 1, self.epoch_num))
        self.target_train_loader.dataset.switch_to_gen_pseudo()  # 切换到伪标签模式（关闭random_mirror）
        self.logger.info("Generate pseudo label...")
        self.model.eval()

        self.model.apply(apply_dropout)

        f_pass = 10 if self.use_uncertainty else 1  # 不确定性计算，前向传播10次。否则前向传播一次

        for image, _, id_label in tqdm_epoch:

            # 预测结果和方差计算
            image = image.to(self.device)
            with torch.no_grad():
                cur_out_prob = []  # 输出的置信度
                for _ in range(f_pass):  # 前向传播 f_pass 次
                    preds, _ = self.model(image)  # 前向传播，返回prediction, feature
                    probs = torch.softmax(preds, dim=1)  # 转换为置信度 b,c,h,w
                    cur_out_prob.append(probs)  #
            if f_pass == 1:
                out_prob = cur_out_prob[0]
                max_value, max_idx = torch.max(out_prob, dim=1)  # 最大的预测值和对应的索引      b,h,w
                max_std = torch.zeros_like(max_value)
            else:
                out_prob = torch.stack(cur_out_prob)  # 当前批次的预测样本堆叠起来 f_pass,b,c,h,w
                out_prob_std = torch.std(out_prob, dim=0)  # 正伪标签的方差 b,c,h,w
                out_prob_mean = torch.mean(out_prob, dim=0)  # 正伪标签的平均置信度 b,c,h,w
                max_value, max_idx = torch.max(out_prob_mean, dim=1)  # 最大的预测值和对应的索引      b,h,w
                max_std = out_prob_std.gather(1, max_idx.unsqueeze(dim=1)).squeeze(1)  # 最大预测值的方差         b,h,w

            # 阈值计算
            thre_conf = []  # 置信度阈值, 长度为num_classes的list
            thre_std = []  # 方差阈值(不确定性阈值)
            for i in range(self.conf['num_classes']):
                mask_i = (max_idx == i)  # 预测为类别 i 的掩码
                num_pixel = max_value[mask_i].size(0)  # 类别 i 的像素数量
                if num_pixel == 0:  # 类别 i 不存在，阈值设置为 0
                    thre_conf.append(0)
                    thre_std.append(0)
                else:
                    mid = num_pixel // 2  # 类别 i 的像素数量的一半
                    half = num_pixel // 4  # 类别 i 的像素数量的四分之一

                    # 类别 i 所有像素点置信度排序后，取中间值和 0.9 两者的最小值作为置信度阈值
                    max_value_i, _ = torch.sort(max_value[mask_i])
                    thre_conf.append(min(0.9, max_value_i[mid]))

                    # 类别 i 所有像素点置信度排序后，取 1/4 值和 0.01 两者的最大值作为方差阈值
                    max_std_i, _ = torch.sort(max_std[mask_i])
                    thre_std.append(max(0.01, max_std_i[half]))

            # 通过切片的操作，计算每个像素点对应 索引 的阈值，例如：某像素点索引为 4, 通过切片操作可以获取 thre_conf 中第4个元素的阈值。
            thre_conf = torch.tensor(thre_conf).cuda()[max_idx].detach()  # (b,h,w)
            thre_std = torch.tensor(thre_std).cuda()[max_idx].detach()  # (b,h,w)

            if self.use_uncertainty:  # 大于置信度阈值，小于不确定性阈值的像素点选择为伪标签
                selected_idx = (max_value >= thre_conf) * (max_std < thre_std)
            else:
                selected_idx = max_value >= thre_conf
            unselected_idx = ~selected_idx  # 取反，选择所以不满足条件的像素点

            pseudo_label = max_idx.clone().cpu()
            pseudo_label[unselected_idx] = self.ignore_index  # 未被选择的像素赋值为ignore_index

            batch_size = max_idx.size(0)
            for b in range(batch_size):
                label = pseudo_label[b]  # 需要保存的伪标签
                output = np.asarray(label, dtype=np.uint8)  # 转换为图片
                output = Image.fromarray(output)

                pseudo_name = id_label[b]  # 标签文件相对路径名
                save_path = self.pseudo_label_dir + pseudo_name  # 伪标签的保存路径

                # 保存伪标签
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                output.save(save_path)

    def updata_target_dataloader(self):  # 更新目标域的训练集
        self.conf['cityscapes']['label_dir'] = self.pseudo_label_dir  # 更新目标域的训练集的标签路径为当前伪标签路径
        self.conf['cityscapes']['use_pseudo'] = True  # 设置为使用伪标签
        self.target_train_loader = get_city_dataloader(self.conf, 'train')

    def train_one_round(self):

        for _ in tqdm(range(self.epoch_each_round), desc="Total {} epochs".format(self.epoch_each_round)):
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
        tqdm_epoch = tqdm(zip(self.source_train_loader, self.target_train_loader), total=self.train_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch + 1, self.epoch_num))
        self.logger.info("")
        self.logger.info("Training on source and target of epoch-{}...".format(self.current_epoch))
        self.eval.reset()  # 混淆矩阵清零
        self.model.train()

        log_dic = {}     # 保存loss的字典
        loss_kwargs = {'norm_order': self.conf['norm_order'],    # 参数字典
                       'thing_type': self.conf['thing_type'],
                       'entropy_type': self.conf['entropy_type'],
                       'centroid_smoothing': self.conf['centroid_smoothing']}

        # 平均指数移动的参数
        for batch_idx, data in enumerate(tqdm_epoch):
            data_s, data_t = data
            self.optimizer.zero_grad()  # 梯度清零

            # 更新学习率
            poly_lr_scheduler(self.optimizer, self.conf['lr'], self.current_iter, self.iter_total, self.conf['poly_power'])
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            #######################
            # Source forward step #
            #######################

            # train data (labeled)
            images, labels, _ = data_s
            images, labels = images.to(self.device), labels.to(device=self.device, dtype=torch.long)  # (b,3,h,w), (b,h,w)

            # 前向传播
            pred_source, feat_source = self.model(images)  # (b c h w)  (b f h' w')
            pred_source_softmax = torch.nn.functional.softmax(pred_source, dim=1)  # (b c h' w')

            # loss计算、反向传播
            cur_loss = self.loss(pred_source, labels)  # 交叉熵损失函数
            cur_loss.backward()  # 反向传播
            log_dic['Source_ce_loss'] = cur_loss.item()

            #######################
            # Target forward step #
            #######################

            # target data (unlabeld)
            images, pseudo_labels, _ = data_t
            images, pseudo_labels = images.to(self.device), pseudo_labels.to(device=self.device, dtype=torch.long)  # (b,3,h,w), (b,h,w)

            # 前向传播
            pred_target, feat_target = self.model(images)  # (b c h w)  (b f h' w')
            pred_target_softmax = torch.nn.functional.softmax(pred_target, dim=1)  # (b c h' w')

            loss_kwargs['source_prob'] = pred_source_softmax.detach()       # 源域预测结果softmax, 源域的所有 tensor 需要detach防止回传到源域的前向传播
            loss_kwargs['target_prob'] = pred_target_softmax                # 目标域预测结果softmax
            loss_kwargs['source_feat'] = feat_source.detach()               # 源域的中间特征
            loss_kwargs['target_feat'] = feat_target                        # 目标域的中间特征
            loss_kwargs['source_label'] = labels.detach()                   # 源域的标签
            loss_kwargs['target_label'] = pseudo_labels                     # 目标域的伪标签

            # 传入参数，计算当前批次的目标域的损失
            loss_dict = self.feat_reg_ST_loss(**loss_kwargs)
            stuff_alignment_loss, thing_alignment_loss, em_loss = loss_dict['stuff_alignment_loss'], loss_dict['thing_alignment_loss'], loss_dict['em_loss']

            thing_alignment_loss = self.lambda_things * thing_alignment_loss
            stuff_alignment_loss = self.lambda_stuff * stuff_alignment_loss
            em_loss = self.lambda_entropy * em_loss

            total_loss = thing_alignment_loss + stuff_alignment_loss + em_loss
            total_loss.backward()

            # 保存当前loss
            log_dic['Stuff_alignment_loss'] = stuff_alignment_loss.item()
            log_dic['Thing_alignment_loss'] = thing_alignment_loss.item()
            log_dic['EM_loss'] = em_loss.item()
            log_dic['Target_loss'] = total_loss.item()
            log_string = 'epoch{}-batch-{}:' + '={:3f}-'.join(log_dic.keys()) + '={:3f}'

            # logging
            if batch_idx % self.conf['iter_show_loss'] == 0:
                self.logger.info(log_string.format(self.current_epoch, batch_idx, *log_dic.values()))
                for name, elem in log_dic.items():
                    self.writer.add_scalar(name, elem, self.current_iter)

            self.current_iter += 1  # 更新总迭代次数

            self.optimizer.step()  # 参数更新

        tqdm_epoch.close()
        self.feat_reg_ST_loss.reset()

        if self.conf['save_inter_model']:
            self.logger.info("Saving model of epoch {} ...".format(self.current_epoch))
            self.save_checkpoint(self.model_name + '_epoch{}.pth'.format(self.current_epoch))

    def validate(self, name='target_val'):
        self.logger.info('Validating on {} of epoch-{}...'.format(name, self.current_epoch))
        self.eval.reset()

        val_loader = self.target_val_loader if name == 'target_val' else self.source_val_loader

        with torch.no_grad():
            tqdm_batch = tqdm(val_loader, desc="Val epoch-{}-".format(self.current_epoch))
            self.model.eval()

            for batch_idx, data in enumerate(tqdm_batch):
                # unpack data
                images, labels, id_labels = data

                b, H, W = labels.shape
                images, labels = images.to(self.device), labels.to(device=self.device, dtype=torch.long)  # (b,3,h,w), (b,H,W) val评估的时候，需要原图大小的标签

                preds, _ = self.model(images)  # prediction, feature
                arg_preds = torch.argmax(preds, dim=1, keepdim=True)  # 预测类别, (b,c,h,w) ==> (b,1,h,w) (要对h，w上采样，interpolate输入必须为4维。因此要keepdim)
                arg_preds = arg_preds.to(torch.float32)  # interpolate 不能处理int类型
                arg_preds = torch.nn.functional.interpolate(arg_preds, size=(H, W), mode='nearest')  # (b,1,h,w) ==> (b,1,H,W) 上采样
                arg_preds = arg_preds.squeeze(dim=1).to(torch.int64)  # (b,1,H,W) ==> (b,H,W), 和labels数据类型一致

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
        except OSError as _:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.conf['checkpoint_dir']))
            self.logger.info("**First time to train**")


def init_config(config_path):
    # Reading configuration file
    config = yaml.load(open(config_path, "r",  encoding='utf-8'), Loader=yaml.FullLoader)
    train_conf = config['train']

    assert train_conf['num_classes'] == 19 if train_conf['source_data_name'] == 'gta5' else train_conf['num_classes'] == 16
    assert train_conf['centroid_smoothing'] <= 1., 'Centroid smoothing coefficient with invalid value: {}'.format(train_conf['centroid_smoothing'])

    # data configure
    source_data_conf = train_conf[train_conf['source_data_name']]
    target_data_conf = train_conf[train_conf['target_data_name']]
    for data_conf in [source_data_conf, target_data_conf]:
        data_conf['size'] = tuple(map(int, data_conf['size'].split(',')))   # 将size的h,w转换为int，并且组合为tuple类型
    if source_data_conf['use_trans_image']:                                 # 使用翻译后的图像
        source_data_conf['image_dir'] = source_data_conf['trans_image_dir']

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
        os.environ["CUDA_VISIBLE_DEVICES"] = train_conf['gpu_id']
        device = torch.device('cuda')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
    # config_path = "config/config_UDA.yaml"
    config_path = opt.conf
    train_conf, logger, device = init_config(config_path)
    agent = Trainer(train_conf, logger, device)
    agent.main()


if __name__ == '__main__':
    main()
