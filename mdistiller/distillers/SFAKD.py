import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


class SFAKD(Distiller):
    def __init__(self, student, teacher, cfg, s_n, t_n, t_cls):
        super().__init__(student, teacher)
        self.feat_s_dim = s_n
        self.feat_t_dim = t_n
        self.t_cls = t_cls

        self.mid_channel = (s_n + t_n) // 2
        
        self.transfer = nn.Sequential(
            nn.Conv2d(self.feat_s_dim, self.mid_channel, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_channel),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.mid_channel, self.mid_channel, kernel_size=3, padding=1, stride=1, bias=False, groups=1),
            # nn.BatchNorm2d(self.mid_channel),
            # nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(self.mid_channel, self.mid_channel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Conv2d(self.mid_channel, self.feat_t_dim, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(self.feat_t_dim),
            nn.ReLU(inplace=True),
        )
        
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #TODO:这个最后要写到train_new.py里去
        if(cfg.DATASET.TYPE == "cifar100"):
            if cfg.DISTILLER.STUDENT[0] == "S" or cfg.DISTILLER.STUDENT[0] == "v":
                shape = 4
            elif cfg.DISTILLER.STUDENT[0] == "M":
                shape = 2
            else:
                shape = 8
        else:
            shape = 7

        self.sfa = SFA_Module(
            in_channels = self.feat_s_dim,
            out_channels = self.feat_s_dim,
            # shapes = 8
            shapes = shape
        )
        
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.transfer.parameters()) + list(self.sfa.parameters())
    
    def get_extra_parameters(self):
        num_transfer = 0
        num_fam = 0
        for p in self.transfer.parameters():
            num_transfer += p.numel()
        print("transfer_parameter:", num_transfer)
        for p in self.sfa.parameters():
            num_fam += p.numel()
        print("sfa_parameter:", num_fam)
        return num_transfer + num_fam
    
    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
        
        feat_student = features_student["feats"]
        feat_teacher = features_teacher["feats"]
        s_H, t_H = feat_student[-1].shape[2], feat_teacher[-1].shape[2]
        # print("s_H:", s_H)
        s_c, t_c = feat_student[-1].shape[1], feat_teacher[-1].shape[1]
        # print("s_c, t_c:", s_c, t_c)
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_student[-1], (t_H, t_H))
            target = feat_teacher[-1]
        else:
            source = feat_student[-1]
            target = F.adaptive_avg_pool2d(feat_teacher[-1], (s_H, s_H))
        
        trans_feat_t = target
        #这里添加一个FAM模块
        source = self.sfa(source)
        # print("source_shape:", source.shape)
        trans_feat_s = self.transfer(source)
        # print("trans_feat_s_shape:", trans_feat_s.shape)
        temp_feat = self.avg_pool(trans_feat_s)
        # print("temp_feat", temp_feat.shape)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        # print("temp_feat", temp_feat.shape)
        pred_feat_s = self.t_cls(temp_feat) #reuse teacher cls head
        
        loss_mse = nn.MSELoss()
        loss_feat = loss_mse(trans_feat_s, trans_feat_t)
        
        loss_ce = F.cross_entropy(pred_feat_s, logits_teacher)
        losses_dict = {
            # "loss_ce": loss_ce,
            "loss_simkd_feat": loss_feat
        }
        return pred_feat_s, losses_dict
    
    def forward_test(self, image):
        with torch.no_grad():
            _, features_student = self.student(image)
            _, features_teacher = self.teacher(image)
            feat_student = features_student["feats"]
            feat_teacher = features_teacher["feats"]
            s_H, t_H = feat_student[-1].shape[2], feat_teacher[-1].shape[2]
            if s_H > t_H:
                source = F.adaptive_avg_pool2d(feat_student[-1], (t_H, t_H))
            else:
                source = feat_student[-1]
            source = self.sfa(source)
            trans_feat_s = self.transfer(source)
            temp_feat = self.avg_pool(trans_feat_s)
            temp_feat = temp_feat.view(temp_feat.size(0), -1)
            pred_feat_s = self.t_cls(temp_feat)

        return pred_feat_s
    

class SFA_Module(nn.Module):
    def __init__(self, in_channels, out_channels, shapes):
        super(SFA_Module, self).__init__()

        """
        feat_s_shape, feat_t_shape
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes
      #  print(self.shapes)
        self.rate1 = torch.nn.Parameter(torch.tensor(0.5))  #频域权重
        self.rate2 = torch.nn.Parameter(torch.tensor(0.5))  #空间域权重
       # self.out_channels = feat_t_shape[1]
        self.scale = (1 / (self.in_channels * self.out_channels))  #初始化权重矩阵的缩放系数
        # weights = nn.Parameter(
        #     self.scale * torch.rand(self.in_channels, self.shapes, 
        #                             self.shapes, dtype=torch.cfloat)
        #                             )
        self.real_weight = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.shapes, self.shapes, dtype=torch.float32)
        )
        self.imag_weight = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.shapes, self.shapes, dtype=torch.float32)
        )
        self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1) #1x1卷积调整维度，用于空间域的特征转换


    def forward(self, x):
        # 1. 傅里叶变换 --------------------------------------------------
        x_ft = torch.fft.fft2(x, norm="ortho")  #对输入进行二维FFT(正交归一化)，形状为(batch_size, in_channels, H, W)

        # 2. 频域卷积 ---------------------------------------------------
        complex_weight = torch.complex(self.real_weight, self.imag_weight)
        out_ft = torch.einsum("bixy,ixy->bixy", x_ft, complex_weight)  #应用可学习的频率滤波器，输出形状为(batch_size, out_channels, H, W)

        h, w = out_ft.shape[2], out_ft.shape[3]  # height and width
        # print(h, w)
        mask = make_gaussian_mask(h, w, 0.1)
        mask = mask.unsqueeze(0).unsqueeze(0).to(out_ft.dtype)

        out_ft = torch.fft.fftshift(out_ft, dim=(-2, -1))  
        out_ft = out_ft * mask 
        out_ft = torch.fft.ifftshift(out_ft, dim=(-2, -1))

        out = torch.fft.ifft2(out_ft, norm="ortho").real  #对频域特征进行逆FFT，得到复数形式的输出

        # 6. 空间域卷积 --------------------------------------------------
        out2 = self.w0(x)  #输出形状为(batch_size, out_channels, H, W)，通过1x1卷积调整维度

        # 7.融合
        return self.rate1 * out + self.rate2*out2

def make_gaussian_mask(h, w, cutoff_ratio, device='cuda'):
        y = torch.arange(h, device=device) - h // 2
        x = torch.arange(w, device=device) - w // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        sigma = cutoff_ratio * min(h, w)
        gaussian = 1 - torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()  # normalize to [0,1]
        return gaussian  # shape [H, W]

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # groups = in_channels 表示深度卷积
            bias=bias
        )
        # 逐点卷积
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        # 可选的批量归一化和激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
