import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class feat_reg_ST_loss(nn.Module):

    def __init__(self, ignore_index=-1, num_class=19, deque_capacity_factor=2.0, feat_channel=2048, device='cuda'):
        super(feat_reg_ST_loss, self).__init__()

        if num_class == 19:  # GTA5
            self.BG_LABEL = [0, 1, 2, 3, 4, 8, 9, 10]
            self.FG_LABEL = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
            # GTA5 标签数量比例
            # self.source_weight = [6305, 617, 2227, 244, 85, 165, 23, 13, 1083, 317, 2110, 25, 6, 493, 224, 66, 14, 6, 1]

            # 特征图大小（2048 161 91），根据特征图大小和标签数量比例计算出的默认容量大小
            self.deque_capacities = [2328, 330, 193, 364, 87, 6949, 3159, 935, 197, 89, 14]

        else:  # SYNTHIA
            self.BG_LABEL = [0, 1, 2, 3, 4, 8, 10]
            self.FG_LABEL = [5, 6, 7, 11, 12, 13, 15, 17, 18]

            # SYNTHIA待计算
            self.deque_capacities = []

        # 像素点数量和权重
        self.source_num_pixel_BG, self.target_num_pixel_BG = [0] * len(self.BG_LABEL), [0] * len(self.BG_LABEL)
        self.source_num_pixel_FG, self.target_num_pixel_FG = [0] * len(self.FG_LABEL), [0] * len(self.FG_LABEL)
        self.source_weight_BG, self.target_weight_BG = [1.] * len(self.BG_LABEL), [1.] * len(self.BG_LABEL)
        self.source_weight_FG, self.target_weight_FG = [1.] * len(self.FG_LABEL), [1.] * len(self.FG_LABEL)

        self.ignore_index = ignore_index
        self.device = device
        self.num_class = num_class
        self.dist_func = None  # L0、L1、L2...

        # 维度
        self.B = None  # batch size
        self.F = feat_channel  # 特征图的通道数
        self.Hs, self.Ws, self.hs, self.ws = None, None, None, None  # 源域预测图和特征图大小
        self.Ht, self.Wt, self.ht, self.wt = None, None, None, None  # 目标域预测图和特征图大小
        self.C_bg, self.C_fg = len(self.BG_LABEL), len(self.FG_LABEL)  # 背景和前景的类别数量

        self.source_feat = None
        self.source_label = None
        self.target_feat = None
        self.target_label = None

        # 背景类：计算源域和目标域的类中心，执行对比学习
        self.centroids_source_BG = None  # 源域背景类中心
        self.centroids_target_BG = None  # 目标域背景类中心
        self.centroids_source_BG_avg = None
        self.centroids_target_BG_avg = None

        # 前景类：计算源域特征向量队列和当前批次目标域特征向量，执行余弦相识度最大化
        self.deque_capacities = [int(capacity * deque_capacity_factor) for capacity in self.deque_capacities]  # 源域特征向量队列容量, 一共有C_fg个队列，每个队列都有自己的容量
        self.remaining_capacities = self.deque_capacities.copy()  # 队列的剩余容量
        self.source_feat_deques = [torch.full(dtype=torch.float32, fill_value=float('Inf'), size=(self.deque_capacities[i], self.F)).to('cuda') for i in range(self.C_fg)]
        self.target_feat_cur = [None] * self.C_fg  # 当前批次目标域特征向量

    def dist(self, tensor, dim=None):
        if isinstance(self.dist_func, int):  # dist_func == 1， L1距离；ist_func == 2， L2距离.....
            return torch.norm(tensor, p=self.dist_func, dim=dim) / tensor.numel()  # sum(abs(tensor)**p)**(1./p) , default output is scalar
        else:
            return self.dist_func(tensor)  # dist_func是距离函数，例如，L1loss(),MSEloss()

    ## PRE-PROCESSING ###
    def feature_processing(self, feat, softmax, label, domain, argmax_dws_type='bilinear'):
        """
        Process feature and softmax tensors in order to get downsampled and 2D-shaped representations
        :param feat: feature tensor (B x F x h x w)
        :param label: Ground truth (B x H x W)
        :param argmax_dws_type: direct 'nearest' or 'bilinear' through from softmax
        """
        self.B = softmax.size(0)
        assert self.F == feat.size(1)
        # 获取源域、目标域的特征图、预测结果的尺度
        if domain == 'source':
            self.Hs, self.Ws = softmax.size(2), softmax.size(3)
            self.hs, self.ws = feat.size(2), feat.size(3)
        else:
            self.Ht, self.Wt = softmax.size(2), softmax.size(3)
            self.ht, self.wt = feat.size(2), feat.size(3)

        # 拉直特征图
        h, w = feat.size(2), feat.size(3)
        feat = feat.permute(0, 2, 3, 1).contiguous()  # size B x h x w x F
        feat = feat.view(-1, feat.size()[-1])  # size N x F (N = B x h x w)

        # 预测结果argmax下采样
        if argmax_dws_type == 'nearest':
            peak_values, argmax = torch.max(softmax, dim=1)  # size B x H x W
            argmax_dws = torch.squeeze(F.interpolate(torch.unsqueeze(argmax.float(), dim=1), size=(h, w), mode='nearest'), dim=1)  # size B x h x w
        else:
            softmax_dws = F.interpolate(softmax, size=(h, w), mode='bilinear', align_corners=True)  # size B x C x h x w
            peak_values_dws, argmax_dws = torch.max(softmax_dws, dim=1)  # size B x h x w

        # 标签下采样
        label_dws = torch.squeeze(F.interpolate(torch.unsqueeze(label.float(), dim=1), size=(h, w), mode='nearest'), dim=1)  # size B x h x w， GT下采样到和特征图一样大

        # 分离计算图
        argmax_dws = argmax_dws.detach()
        label_dws = label_dws.detach()

        # 标签处理
        if domain == 'source':  # 源域只计算预测正确的标签
            label_dws[label_dws != argmax_dws] = self.ignore_index  # 忽略预测错误
            # pass
        else:  # 目标域补全伪标签
            label_dws_ignore_mask = torch.eq(label_dws, self.ignore_index)  # 伪标签为空的mask
            label_dws[label_dws_ignore_mask] = argmax_dws.to(torch.float32)[label_dws_ignore_mask]  # 用预测结果补充伪标签的空缺

        label_dws = label_dws.view(-1)  # size N（N = B x h x w）

        if domain == 'source':
            self.source_feat = feat.detach()  # N x F, 每一个点的特征向量，需要detach防止回传到源域的前向传播
            self.source_label = label_dws  # N , 源域预测正确标签
        else:
            self.target_feat = feat  # N x F, 每一个点的特征向量
            self.target_label = label_dws  # N,  目标域补全伪标签

    def computer_stuff(self, centroids_smoothing=-1):
        """
        背景类：计算源域和目标域的类中心，执行对比学习
        :param centroids_smoothing: if > 0 new centroids are updated over avg past ones
        """
        centroid_list_source_BG, centroid_list_target_BG = [], []  # 保存每一个类的平均特征向量，列表大小C_BG(背景类的数量)

        # 计算源域和目标域的背景（stuff）的中心
        for i, label_i in enumerate(self.BG_LABEL):
            # boolean tensor, True where features belong to class label_i
            source_mask = torch.eq(self.source_label.detach(), label_i)  # size N
            target_mask = torch.eq(self.target_label.detach(), label_i)  # size N

            # select only features of class label_i
            source_feat_i = self.source_feat[source_mask, :]  # size Ns_i x F
            target_feat_i = self.target_feat[target_mask, :]  # size Nt_i x F

            # 更新前景像素数量
            self.source_num_pixel_BG[i] += source_feat_i.size(0)
            self.target_num_pixel_BG[i] += target_feat_i.size(0)

            # compute the source centroid of class label_i
            if source_feat_i.size(0) > 0:  # class label_i点的数量大于0
                centroid = torch.mean(source_feat_i, dim=0, keepdim=True)  # size 1 x F，计算平均特征向量
                centroid_list_source_BG.append(centroid)
            else:  # class label_i不存在
                centroid = torch.tensor([[float("Inf")] * self.F], dtype=torch.float).to(self.device)  # size 1 x F
                centroid_list_source_BG.append(centroid)

            # compute the target centroid of class label_i
            if target_feat_i.size(0) > 0:
                centroid = torch.mean(target_feat_i, dim=0, keepdim=True)  # size 1 x F
                centroid_list_target_BG.append(centroid)
            else:
                centroid = torch.tensor([[float("Inf")] * self.F], dtype=torch.float).to(self.device)  # size 1 x F
                centroid_list_target_BG.append(centroid)

        self.centroids_source_BG = torch.squeeze(torch.stack(centroid_list_source_BG, dim=0))  # size C_BG x 1 x F -> C_BG x F
        self.centroids_target_BG = torch.squeeze(torch.stack(centroid_list_target_BG, dim=0))  # size C_BG x 1 x F -> C_BG x F

        # 类别平衡系数
        # oc = Nc/N , oc小于1/5的都替换为1/5
        # Wc = min(N/Nc，μ）/ SUM(min(N/Nc，μ)
        # 1/oc = min(N/Nc，μ）
        source_N = sum(self.source_num_pixel_BG)
        target_N = sum(self.target_num_pixel_BG)
        #
        source_Oc_inverse = [min(source_N / float(Nc), 10) if Nc != 0 else 10 for Nc in self.source_num_pixel_BG]
        target_Oc_inverse = [min(target_N / float(Nc), 10) if Nc != 0 else 10 for Nc in self.target_num_pixel_BG]
        source_Oc_inverse_sum = sum(source_Oc_inverse)
        target_Oc_inverse_sum = sum(target_Oc_inverse)
        for i in range(self.C_bg):
            self.source_weight_BG[i] = source_Oc_inverse[i] / float(source_Oc_inverse_sum)
            self.target_weight_BG[i] = target_Oc_inverse[i] / float(target_Oc_inverse_sum)

        # 指数平均移动
        if centroids_smoothing >= 0.:
            if self.centroids_source_BG_avg is None: self.centroids_source_BG_avg = self.centroids_source_BG  # size C_BG x F
            # In early steps there may be no centroids for small classes, so avoid averaging with Inf values by replacing them with values of current step
            self.centroids_source_BG_avg = torch.where(self.centroids_source_BG_avg != float('inf'), self.centroids_source_BG_avg, self.centroids_source_BG)
            # In some steps there may be no centroids for some classes, so avoid averaging with Inf values by replacing them with avg values
            self.centroids_source_BG = torch.where(self.centroids_source_BG == float('inf'), self.centroids_source_BG_avg.detach(), self.centroids_source_BG)
            # Exponential Moving Average
            self.centroids_source_BG = centroids_smoothing * self.centroids_source_BG + (1 - centroids_smoothing) * self.centroids_source_BG_avg.detach()
            self.centroids_source_BG_avg = self.centroids_source_BG.detach().clone()

            if self.centroids_target_BG_avg is None: self.centroids_target_BG_avg = self.centroids_target_BG  # size C_BG x F
            self.centroids_target_BG_avg = torch.where(self.centroids_target_BG_avg != float('inf'), self.centroids_target_BG_avg, self.centroids_target_BG)
            self.centroids_target_BG = torch.where(self.centroids_target_BG == float('inf'), self.centroids_target_BG_avg.detach(), self.centroids_target_BG)
            self.centroids_target_BG = centroids_smoothing * self.centroids_target_BG + (1 - centroids_smoothing) * self.centroids_target_BG_avg.detach()
            self.centroids_target_BG_avg = self.centroids_target_BG.detach().clone()

    def computer_things(self):

        for i, label_i in enumerate(self.FG_LABEL):

            source_mask = torch.eq(self.source_label.detach(), label_i)  # size N，类别label_i的掩码
            target_mask = torch.eq(self.target_label.detach(), label_i)  # size N

            # select only features of class label_i
            source_feat_i = self.source_feat[source_mask, :]  # size Ns_i x F
            target_feat_i = self.target_feat[target_mask, :]  # size Nt_i x F

            # 更新背景像素数量
            self.source_num_pixel_FG[i] += source_feat_i.size(0)
            self.target_num_pixel_FG[i] += target_feat_i.size(0)

            num_source_feat_i = source_feat_i.size(0)  # 源域类别label_i的特征向量数量
            if num_source_feat_i > 0:
                if num_source_feat_i <= self.deque_capacities[i]:
                    self.source_feat_deques[i] = torch.cat((self.source_feat_deques[i][num_source_feat_i:], source_feat_i))  # 队列可以放入label_i的特征向量, 用cat方法实现FIFO操作
                else:
                    self.source_feat_deques[i] = source_feat_i[:self.deque_capacities[i]]  # 队列放不下label_i的特征向量，放入部分label_i的特征向量

                self.remaining_capacities[i] = max(0, self.remaining_capacities[i] - num_source_feat_i)  # 更新剩余容量

            self.target_feat_cur[i] = target_feat_i

        # 类别平衡系数
        # Wc = min(N/Nc，μ）/ SUM(min(N/Nc，μ)
        # 1/oc = min(N/Nc，μ）
        source_N = sum(self.source_num_pixel_FG)
        target_N = sum(self.target_num_pixel_FG)
        source_Oc_inverse = [min(source_N / float(Nc), 10) if Nc != 0 else 10 for Nc in self.source_num_pixel_FG]   # 防止Nc过小，以及防止Nc为0
        target_Oc_inverse = [min(target_N / float(Nc), 10) if Nc != 0 else 10 for Nc in self.target_num_pixel_FG]
        source_Oc_inverse_sum = sum(source_Oc_inverse)
        target_Oc_inverse_sum = sum(target_Oc_inverse)
        for i in range(self.C_fg):
            self.source_weight_FG[i] = source_Oc_inverse[i] / float(source_Oc_inverse_sum)
            self.target_weight_FG[i] = target_Oc_inverse[i] / float(target_Oc_inverse_sum)

    def stuff_alignment(self, T=1):
        # 源域和目标域背景类中心
        centroids_source = self.centroids_source_BG  # size：C_bg * F
        centroids_target = self.centroids_target_BG  # size：C_bg * F

        # 计算有效类中心索引
        seen_source_indices = [i for i in range(self.C_bg) if not torch.isnan(centroids_source[i, 0]) and not centroids_source[i, 0] == float('Inf')]  # list of C_bg elems, True for seen classes, False elsewhere
        seen_target_indices = [i for i in range(self.C_bg) if not torch.isnan(centroids_target[i, 0]) and not centroids_target[i, 0] == float('Inf')]  # list of C_bg elems, True for seen classes, False elsewhere

        # 计算源域和目标域共同的有效类中心索引
        seen_source_target_indices = [i for i in seen_source_indices if i in seen_target_indices]

        # 源域和目标域的背景类中心归一化
        centroids_source_nor = F.normalize(centroids_source[seen_source_indices], dim=1)  # size：C_seen_s * F
        centroids_target_nor = F.normalize(centroids_target[seen_target_indices], dim=1)  # size：C_seen_t * F

        # 计算目标域和源域类中心相似度矩阵
        sim_matrix = torch.mm(centroids_target_nor, centroids_source_nor.t())  # C_seen_t * C_seen_s
        sim_matrix = sim_matrix / T
        CL_loss = torch.tensor([0.]).cuda()

        for i in seen_source_target_indices:

            i_to_row = seen_target_indices.index(i)  # 索引i转换为相似度矩阵的行
            i_to_col = seen_source_indices.index(i)  # 索引i转换为相似度矩阵的列

            indices_but_i = np.array([ind for ind in seen_source_indices if ind != i])  # 索引i以外的索引（负例索引）
            if len(indices_but_i) > 0:  # 只计算有负例的情况
                # 正例相似度
                sim_pos = torch.exp(sim_matrix[i_to_row, i_to_col])
                # 负例相似度
                sim_neg = torch.tensor([0.]).cuda()
                for ind in indices_but_i:
                    ind_to_col = seen_source_indices.index(ind)  # 索引ind转换为相似度矩阵的列
                    sim_neg += torch.exp(sim_matrix[i_to_row, ind_to_col])

                # 更新对比学习损失(带权重)
                weight = self.target_weight_BG[i]  # 索引i(类别i)的权重
                CL_loss += -torch.log(sim_pos / sim_neg) * weight
                # CL_loss += -torch.log(sim_pos / (sim_neg + sim_pos)) * weight
        CL_loss /= len(seen_source_target_indices)
        return CL_loss

    def things_alignment(self, T_cos=0.5, loss_type='entropy'):

        # 源域特征向量队列和目标域特征向量
        source_feat_deques = self.source_feat_deques  # 列表结构，列表长度：C_fg，每个元素的大小： deque_capacities_i * F
        target_feat = self.target_feat_cur  # 列表结构，列表长度：C_fg，每个元素的大小： Nt_i * F

        sim_loss = torch.tensor([0.]).cuda()
        for i in range(self.C_fg):
            # 过滤掉源域特征向量队列的空元素
            source_feat_i = source_feat_deques[i][self.remaining_capacities[i]:]  # 源域特征向量队列，deque_len_i * F
            target_feat_i = target_feat[i]  # 目标域特征向量，  Nt_i * F
            weight = self.target_weight_FG[i]  # 索引i(类别i)的权重

            if source_feat_i.size(0) > 0 and target_feat_i.size(0) > 0:
                # 有三种loss
                if loss_type == 'Entropy':  # 最大化entropy：-p *log(p)
                    matrix = torch.mm(target_feat_i, source_feat_i.t())  # size: Nt_i * deque_len_i
                    matrix_softmax = torch.softmax(matrix, dim=1)
                    # matrix_log_softmax = torch.log_softmax(matrix, dim=1)
                    sim_loss += -torch.mean((matrix_softmax * torch.log2(matrix_softmax+1e-30))) * weight  # matrix_log_softmax替换torch.log(matrix_softmax)可以防止溢出

                if loss_type == 'Squares':  # 最大化：1-p^2
                    matrix = torch.mm(target_feat_i, source_feat_i.t())  # size: Nt_i * deque_len_i
                    matrix_softmax = torch.softmax(matrix, dim=1)
                    sim_loss += torch.mean(1 - torch.pow(matrix_softmax, 2)) * weight

                if loss_type == 'Cosine':  # ③最大化余弦相识度
                    # 归一化
                    source_feat_nor_i = F.normalize(source_feat_i, dim=1)  # size: deque_capacities_i * F
                    target_feat_nor_i = F.normalize(target_feat_i, dim=1)  # size: Nt_i * F

                    # 余弦相识度
                    cos_sim_matrix = torch.mm(target_feat_nor_i, source_feat_nor_i.t())  # size: Nt_i * deque_capacities_i

                    # 选择大于阈值的元素
                    select_mask = torch.gt(cos_sim_matrix, T_cos)

                    # 大于阈值的元素的数量大于0，计算loss
                    if select_mask[select_mask == True].size(0) > 0:
                        cos_sim_matrix_select = cos_sim_matrix[select_mask]
                        sim_loss += torch.mean(1. - cos_sim_matrix_select) * weight

        return sim_loss

    def entropy_loss(self, pred_softmax, label, loss_type='Squares'):

        max_pred, arg_pred = torch.max(pred_softmax, dim=1)  # values, indices  (N, H, W)
        label_ignore_mask = torch.eq(label.detach(), self.ignore_index)  # 伪标签为空的mask, (N, H, W)
        label[label_ignore_mask] = arg_pred[label_ignore_mask]  # 用预测结果补充伪标签的空缺
        mask = (arg_pred == label)  # (N, H, W)

        weights = [0.] * self.num_class
        for i, idx in enumerate(self.BG_LABEL):
            weights[idx] = self.target_weight_BG[i]
        for i, idx in enumerate(self.FG_LABEL):
            weights[idx] = self.target_weight_FG[i]

        weights = torch.tensor(weights).cuda()[arg_pred].detach()  # (num_class) ==> (N, 1, H, W)
        weights = weights.expand_as(pred_softmax)  # (N, 1, H, W) ==> (N, C, H, W)

        if loss_type == 'Squares':  # 1 - p^2
            entropy_loss = torch.mean((1 - (torch.pow(pred_softmax, 2)) * weights))
        if loss_type == 'Entropy':  # -p * log (p)
            entropy_loss = -torch.mean((pred_softmax * torch.log2(pred_softmax+1e-30) * weights))
        if loss_type == 'FocalEntropy':  # -p * log (p) * (1 - p)^2
            entropy_loss = -torch.mean((pred_softmax * torch.log2(pred_softmax+1e-30) * torch.pow(1 - pred_softmax, 2) * weights))

        return entropy_loss

    def forward(self, **kwargs):

        # 特征处理
        # 传入特征图、预测结果预测结果softmax（置信度）、GT、源域还是目标域
        # 计算self.source_feat、self.source_label
        alignment_params = kwargs.get('alignment_params')
        norm_order = alignment_params['norm_order']
        thing_type = alignment_params['thing_type']
        em_type = alignment_params['em_type']
        self.dist_func = norm_order

        self.feature_processing(feat=kwargs.get('source_feat'), softmax=kwargs.get('source_prob'), label=kwargs.get('source_label'), domain='source')
        self.feature_processing(feat=kwargs.get('target_feat'), softmax=kwargs.get('target_prob'), label=kwargs.get('target_label'), domain='target')

        smo_coeff = kwargs['smo_coeff']
        assert smo_coeff <= 1., 'Centroid smoothing coefficient with invalid value: {}'.format(smo_coeff)
        self.computer_things()
        self.computer_stuff(centroids_smoothing=smo_coeff)
        stuff_alignment_loss = self.stuff_alignment()
        thing_alignment_loss = self.things_alignment(loss_type=thing_type)
        EM_loss = self.entropy_loss(pred_softmax=kwargs.get('target_prob'), label=kwargs.get('target_label'), loss_type=em_type)
        output = {'stuff_alignment_loss': stuff_alignment_loss, 'thing_alignment_loss': thing_alignment_loss, 'EM_loss': EM_loss}
        return output

    def reset(self):
        # 每个epoch后重置关键变量
        self.source_num_pixel_BG, self.target_num_pixel_BG = [0] * len(self.BG_LABEL), [0] * len(self.BG_LABEL)
        self.source_num_pixel_FG, self.target_num_pixel_FG = [0] * len(self.FG_LABEL), [0] * len(self.FG_LABEL)
        self.source_weight_BG, self.target_weight_BG = [1.] * len(self.BG_LABEL), [1.] * len(self.BG_LABEL)
        self.source_weight_FG, self.target_weight_FG = [1.] * len(self.FG_LABEL), [1.] * len(self.FG_LABEL)

        self.centroids_source_BG = None  # 源域背景类中心
        self.centroids_target_BG = None  # 目标域背景类中心
        self.centroids_source_BG_avg = None
        self.centroids_target_BG_avg = None

        self.remaining_capacities = self.deque_capacities.copy()  # 队列的剩余容量
        self.source_feat_deques = [torch.full(dtype=torch.float32, fill_value=float('Inf'), size=(self.deque_capacities[i], self.F)).to('cuda') for i in range(self.C_fg)]
        self.target_feat_cur = [None] * self.C_fg  # 当前批次目标域特征向量


if __name__ == '__main__':

    ignore_index = -1
    num_class = 19
    feat_channel = 1024
    batch_size = 1
    torch.manual_seed(3)
    feature_s = torch.randint(0, 10, (batch_size, feat_channel, 128, 128), dtype=torch.float, requires_grad=True).cuda()  # (B  F  h  w)
    feature_t = torch.randint(0, 10, (batch_size, feat_channel, 128, 128), dtype=torch.float, requires_grad=True).cuda()  # (B  F  h  w)
    s_pred = torch.randn((batch_size, num_class, 256, 256), dtype=torch.float, requires_grad=True).cuda()  # (B  C  H  W)
    t_pred = torch.randn((batch_size, num_class, 256, 256), dtype=torch.float, requires_grad=True).cuda()  # (B  C  H  W)

    print(torch.mean(feature_s))
    s_softmax = torch.nn.functional.softmax(s_pred, dim=1)  # (B  C  H  W)
    t_softmax = torch.nn.functional.softmax(t_pred, dim=1)  # (B  C  H  W)

    _, s_pred_y = torch.max(s_softmax, dim=1)
    _, t_pred_y = torch.max(t_softmax, dim=1)

    s_y = torch.randint(0, num_class, (batch_size, 256, 256), dtype=torch.long).cuda()  # (B  H  W)
    t_py = torch.randint(0, num_class, (batch_size, 256, 256), dtype=torch.long).cuda()  # (B  H  W)

    # 模拟伪标签的空缺
    random = torch.randint(0, 4, (batch_size, 256, 256), dtype=torch.long).cuda()
    mask_select = (random == 1)
    t_py[mask_select] = ignore_index

    # 模拟相等的标签
    random_s = torch.randint(0, 3, (batch_size, 256, 256), dtype=torch.long).cuda()
    mask_select = (random_s != 0)
    s_y[mask_select] = s_pred_y[mask_select]

    random_t = torch.randint(0, 3, (batch_size, 256, 256), dtype=torch.long).cuda()
    mask_select = (random_t != 0)
    t_py[mask_select] = t_pred_y[mask_select]

    f = feat_reg_ST_loss(feat_channel=feat_channel)
    loss_kwargs = {}
    alignment_params = {'norm_order': 1}
    loss_kwargs['alignment_params'] = alignment_params

    loss_kwargs['source_prob'] = s_softmax
    loss_kwargs['target_prob'] = t_softmax
    loss_kwargs['source_feat'] = feature_s
    loss_kwargs['target_feat'] = feature_t
    loss_kwargs['source_label'] = s_y
    loss_kwargs['target_label'] = t_py
    loss_kwargs['smo_coeff'] = 0.9
    loss = f(**loss_kwargs)
    print(loss)
