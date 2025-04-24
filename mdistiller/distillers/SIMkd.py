import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


class SimKD(Distiller):
    def __init__(self, student, teacher, cfg, s_n, t_n, t_cls):
        super().__init__(student, teacher)
        self.feat_s_dim = s_n
        self.feat_t_dim = t_n
        self.t_cls = t_cls
        self.SIM_weight = cfg.SIMKD.LOSS.FEAT_WEIGHT
        self.factor = cfg.SIMKD.FACTOR
        
        self.transfer = nn.Sequential(
            nn.Conv2d(self.feat_s_dim, self.feat_t_dim // self.factor, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(self.feat_t_dim // self.factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_t_dim // self.factor, self.feat_t_dim // self.factor, kernel_size=3, padding=1, stride=1, bias=False, groups=1),
            nn.BatchNorm2d(self.feat_t_dim // self.factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_t_dim // self.factor, self.feat_t_dim, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(self.feat_t_dim),
            nn.ReLU(inplace=True),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.transfer.parameters())
    
    def get_extra_parameters(self):
        num_p = 0
        for p in self.transfer.parameters():
            num_p += p.numel()
        return num_p
    
    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
        
        feat_student = features_student["feats"]
        feat_teacher = features_teacher["feats"]
        s_H, t_H = feat_student[-1].shape[2], feat_teacher[-1].shape[2]
        # print(s_H, t_H)
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_student[-1], (t_H, t_H))
            target = feat_teacher[-1]
        else:
            source = feat_student[-1]
            target = F.adaptive_avg_pool2d(feat_teacher[-1], (s_H, s_H))
        
        trans_feat_t = target
        trans_feat_s = self.transfer(source)
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = self.t_cls(temp_feat) #reuse teacher cls head
        
        loss_mse = nn.MSELoss()
        loss_feat = loss_mse(trans_feat_s, trans_feat_t) * self.SIM_weight
        
        loss_ce = F.cross_entropy(pred_feat_s, logits_teacher)
        losses_dict = {
            # "loss_ce": loss_ce,
            "loss_simkd_feat": loss_feat
        }
        return pred_feat_s, losses_dict
    
    def forward_test(self, image):
        with torch.no_grad():
            _, features_student = self.student(image)
            feat_student = features_student["feats"]
            source = feat_student[-1]
            trans_feat_s = self.transfer(source)
            temp_feat = self.avg_pool(trans_feat_s)
            temp_feat = temp_feat.view(temp_feat.size(0), -1)
            pred_feat_s = self.t_cls(temp_feat)

        return pred_feat_s