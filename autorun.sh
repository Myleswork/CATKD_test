# python train.py --cfg configs/cifar100/CAT_KD/vgg13_vgg8.yaml
# python train.py --cfg configs/cifar100/CAT_KD/vgg13_vgg8.yaml
# python train.py --cfg configs/cifar100/CAT_KD/res32x4_shuv1.yaml
# python train.py --cfg configs/cifar100/CAT_KD/res32x4_shuv1.yaml
# python train.py --cfg configs/cifar100/CAT_KD/res32x4_shuv2.yaml
# python train.py --cfg configs/cifar100/CAT_KD/res32x4_shuv2.yaml
# python train.py --cfg configs/cifar100/CAT_KD/res56_res20.yaml
# python train.py --cfg configs/cifar100/CAT_KD/res56_res20.yaml
# python train.py --cfg configs/cifar100/CAT_KD/vgg13_mv2.yaml
# python train.py --cfg configs/cifar100/CAT_KD/vgg13_mv2.yaml
# python train.py --cfg configs/cifar100/CAT_KD/wrn_40_2_wrn_16_2.yaml
# python train.py --cfg configs/cifar100/CAT_KD/wrn_40_2_wrn_16_2.yaml
# python train.py --cfg configs/cifar100/CAT_KD/wrn_40_2_wrn_40_1.yaml
# python train.py --cfg configs/cifar100/CAT_KD/wrn_40_2_wrn_40_1.yaml
# python train.py --cfg configs/cifar100/CAT_KD/wrn40_2_shuv1.yaml
# python train.py --cfg configs/cifar100/CAT_KD/wrn40_2_shuv1.yaml
# python train.py --cfg configs/cifar100/CAT_KD/res32x4_res8x4.yaml
python train.py --cfg configs/cifar100/CAT_KD/res32x4_res8x4.yaml DISTILLER.TYPE "CAT_hcl_KD" CAT_KD.LOSS.CAT_loss_weight 10.
python train.py --cfg configs/cifar100/CAT_KD/res32x4_res8x4.yaml DISTILLER.TYPE "CAT_hcl_KD" CAT_KD.LOSS.CAT_loss_weight 20.