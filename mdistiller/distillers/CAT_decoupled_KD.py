import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
    
class CAT_TEST_KD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(CAT_TEST_KD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CAT_KD.LOSS.CE_WEIGHT
        self.CAT_loss_weight = cfg.CAT_KD.LOSS.CAT_loss_weight
        self.DE_loss_weight = cfg.CAT_KD.LOSS.DE_LOSS_WEIGHT
        self.onlyCAT = cfg.CAT_KD.onlyCAT
        self.CAM_RESOLUTION = cfg.CAT_KD.LOSS.CAM_RESOLUTION
        self.relu = nn.ReLU()
        
        self.IF_NORMALIZE = cfg.CAT_KD.IF_NORMALIZE
        self.IF_BINARIZE = cfg.CAT_KD.IF_BINARIZE
        
        self.IF_OnlyTransferPartialCAMs = cfg.CAT_KD.IF_OnlyTransferPartialCAMs
        self.CAMs_Nums = cfg.CAT_KD.CAMs_Nums
        # 0: select CAMs with top x predicted classes
        # 1: select CAMs with the lowest x predicted classes
        self.Strategy = cfg.CAT_KD.Strategy
        
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)       
        tea = feature_teacher["feats"][-1]
        stu = feature_student["feats"][-1]

        loss_feat = self.CAT_loss_weight * CAT_loss(
            stu, tea, self.CAM_RESOLUTION, self.IF_NORMALIZE
        )

        loss_feat_de = self.DE_loss_weight * DCAMs_Loss(
            stu, tea, self.CAM_RESOLUTION, self.IF_NORMALIZE, target
        )
         
        if self.onlyCAT is False:
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            losses_dict = {
                "loss_CE": loss_ce,
                "loss_CAT": loss_feat,
                "loss_CAT_de": loss_feat_de,
            }
        else:
            losses_dict = {
                "loss_CAT": loss_feat,
                "loss_CAT_de": loss_feat_de,
            }

        return logits_student, losses_dict


def _Normalize(feat,IF_NORMALIZE):
    if IF_NORMALIZE:
        feat = F.normalize(feat,dim=(2,3))
    return feat

def CAT_loss(CAM_Student, CAM_Teacher, CAM_RESOLUTION, IF_NORMALIZE): 
    CAM_Student = F.adaptive_avg_pool2d(CAM_Student, (CAM_RESOLUTION, CAM_RESOLUTION))
    CAM_Teacher = F.adaptive_avg_pool2d(CAM_Teacher, (CAM_RESOLUTION, CAM_RESOLUTION))
    loss = F.mse_loss(_Normalize(CAM_Student, IF_NORMALIZE), _Normalize(CAM_Teacher, IF_NORMALIZE))
    # print("loss", loss)
    return loss

def DCAMs_Loss(CAM_Student, CAM_Teacher, CAM_RESOLUTION, IF_NORMALIZE, target, lambda_target=8, lambda_non_target=1):
    """
    Decoupled CAMs Loss: 分解 CAMs 为目标类和非目标类，分别计算损失。

    :param CAM_Student: (batch_size, num_classes, H, W) - 学生模型的 CAMs
    :param CAM_Teacher: (batch_size, num_classes, H, W) - 教师模型的 CAMs
    :param gt_label: (batch_size,) - 每个样本的 ground truth 类别索引
    :param lambda_target: 目标类 CAM loss 的权重
    :param lambda_non_target: 非目标类 CAM loss 的权重
    :return: 总的 CAMs Loss
    """
    # print(target)
    # CAM_Student = F.adaptive_avg_pool2d(CAM_Student, (CAM_RESOLUTION, CAM_RESOLUTION))
    # CAM_Teacher = F.adaptive_avg_pool2d(CAM_Teacher, (CAM_RESOLUTION, CAM_RESOLUTION))

    if False:   
        CAM_Student = _Normalize(CAM_Student, IF_NORMALIZE)
        CAM_Teacher = _Normalize(CAM_Teacher, IF_NORMALIZE)

    batch_size, num_classes, H, W = CAM_Student.shape #[batch, nums, H, W]
    
    
    # 计算目标类 CAMs
    CAM_Student_target = CAM_Student[torch.arange(batch_size), target]  # (batch_size, H, W)  (64, 8, 8)
    CAM_Teacher_target = CAM_Teacher[torch.arange(batch_size), target]  # (batch_size, H, W)

    #计算非目标类 CAMs
    all_classes = torch.arange(num_classes, device=CAM_Student.device).unsqueeze(0).expand(batch_size, -1)  # 生成类别索引 (batch_size, num_classes)
    non_target_mask = all_classes != target.unsqueeze(1)  # (batch_size, num_classes)
    CAM_Student_non_target = CAM_Student[non_target_mask].view(batch_size, num_classes-1, H, W)  # (batch_size, num_classes-1, H, W)
    CAM_Teacher_non_target = CAM_Teacher[non_target_mask].view(batch_size, num_classes-1, H, W)  # (batch_size, num_classes-1, H, W)
    CAM_Student_non_target_sum = torch.sum(CAM_Student_non_target, dim=1, keepdim=True) # (batch_size, 1, H, W)
    CAM_Teacher_non_target_sum = torch.sum(CAM_Teacher_non_target, dim=1, keepdim=True) # (batch_size, 1, H, W)

    CAM_CONCAT_stu = torch.cat([CAM_Student_target.unsqueeze(1), CAM_Student_non_target_sum], dim=1) # (batch_size, 2, H, W) 用于计算TCKD
    CAM_CONCAT_tea = torch.cat([CAM_Teacher_target.unsqueeze(1), CAM_Teacher_non_target_sum], dim=1) # (batch_size, 2, H, W) 用于计算TCKD

    if True:
        CAM_CONCAT_stu = F.normalize(CAM_CONCAT_stu, dim=(2,3))
        CAM_CONCAT_tea = F.normalize(CAM_CONCAT_tea, dim=(2,3))

    if True:
        CAM_Student_non_target = F.normalize(CAM_Student_non_target, dim=(2,3))
        CAM_Teacher_non_target = F.normalize(CAM_Teacher_non_target, dim=(2,3))
    
    # 计算损失
    loss_target = F.mse_loss(CAM_CONCAT_stu, CAM_CONCAT_tea)
    loss_non_target = F.mse_loss(CAM_Student_non_target, CAM_Teacher_non_target)

    # 组合损失
    total_loss = lambda_target * loss_target + lambda_non_target * loss_non_target
    return total_loss