import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg
from thop import profile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_type = "wrn_40_4"
model_dir = '/home/zhao/project/CAT-KD/download_ckpts/cifar_teachers/wrn_40_4_vanilla/student_best'
#"cifar100", "imagenet"
dataset = "cifar100"
cfg.DATASET.TYPE = "cifar100"
cfg.DATASET.TEST.BATCH_SIZE = 512
if dataset == "imagenet":
    val_loader = get_imagenet_val_loader(64)
    if model_dir == "pretrain":
        model = imagenet_model_dict[model_type](pretrained=True)
    else:
        model = imagenet_model_dict[model_type](pretrained=False)
        model.load_state_dict(load_checkpoint(model_dir)["model"])
elif dataset == "cifar100":
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model, pretrain_model_path = cifar_model_dict[model_type]
    model = model(num_classes=num_classes)
    ckpt = pretrain_model_path if model_dir == "pretrain" else model_dir
    weights = load_checkpoint(ckpt)["model"]
    
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    weights = weights_dict
    
    model.load_state_dict(weights)
model.eval()
model = Vanilla(model)
model = model.cuda()

input_size = (1, 3, 32, 32) if dataset == "cifar100" else (1, 3, 224, 224)
input = torch.randn(input_size).cuda()
flops, params = profile(model, inputs=(input,))

logger.info(f"Number of model parameters: {params/1e6:.2f}M")
logger.info(f"FLOPs: {flops/1e9:.2f}G")

model = torch.nn.DataParallel(model)
test_acc, test_acc_top5, test_loss = validate(val_loader, model)