import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def make_gaussian_mask(h, w, cutoff_ratio=0.1, device='cpu'):
    y = torch.arange(h, device=device) - h // 2
    x = torch.arange(w, device=device) - w // 2
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sigma = cutoff_ratio * min(h, w)
    mask = 1 - torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return mask / mask.max()

class SFA_Module(nn.Module):
    def __init__(self, channels, shapes=8, cuton=0.1):
        super().__init__()
        self.rate1 = nn.Parameter(torch.tensor(0.5))
        self.rate2 = nn.Parameter(torch.tensor(0.5))
        self.weights = nn.Parameter(
            (1/(channels*channels)) * torch.rand(channels, shapes, shapes, dtype=torch.cfloat)
        )
        self.w0 = nn.Conv2d(channels, channels, 1, bias=False)
        self.cutoff = cuton

    def forward(self, x):
        # frequency branch
        x_ft = torch.fft.fft2(x, norm='ortho')
        out_ft = torch.einsum('bixy,ixy->bixy', x_ft, self.weights)
        out_ft = torch.fft.fftshift(out_ft)
        h, w = out_ft.shape[-2:]
        mask = make_gaussian_mask(h, w, self.cutoff, x.device)
        out_ft = out_ft * mask.unsqueeze(0).unsqueeze(0)
        out_ft = torch.fft.ifftshift(out_ft)
        freq_out = torch.fft.ifft2(out_ft, norm='ortho').real
        # spatial branch
        spat_out = self.w0(x)
        return self.rate1 * freq_out + self.rate2 * spat_out

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.point(self.depth(x))))

class SimKD(Distiller):
    def __init__(self, student, teacher, cfg, s_n, t_n, t_cls):
        super().__init__(student, teacher)
        if(cfg.DATASET.TYPE == "cifar100"):
            if cfg.DISTILLER.STUDENT[0] == "S" or cfg.DISTILLER.STUDENT[0] == "v":
                shape = 4
            elif cfg.DISTILLER.STUDENT[0] == "M":
                shape = 2
            else:
                shape = 8
        self.t_cls = t_cls
        self.weight = cfg.SIMKD.LOSS.FEAT_WEIGHT
        self.sfa = SFA_Module(s_n, shapes=shape, cuton=cfg.SIMKD.CUTOFF)
        mid = (s_n + t_n) // 2
        self.transfer = nn.Sequential(
            nn.Conv2d(s_n, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(mid, mid),
            nn.Conv2d(mid, t_n, 1, bias=False),
            nn.BatchNorm2d(t_n), nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_train(self, img, target=None, **kw):
        pred_s, feat_s = self.student(img)
        with torch.no_grad():
            _, feat_t = self.teacher(img)
        fs = feat_s['feats'][-1]
        ft = feat_t['feats'][-1]
        sh, th = fs.size(2), ft.size(2)
        if sh != th:
            fs = F.adaptive_avg_pool2d(fs, (min(sh,th),)*2)
            ft = F.adaptive_avg_pool2d(ft, (min(sh,th),)*2)
        x = self.sfa(fs)
        x = self.transfer(x)
        p = self.t_cls(self.pool(x).view(x.size(0), -1))
        loss = F.mse_loss(x, ft) * self.weight
        return p, {'loss_feat': loss}

    def forward_test(self, img):
        with torch.no_grad():
            _, feat_s = self.student(img)
            x = self.transfer(self.sfa(feat_s['feats'][-1]))
            return self.t_cls(self.pool(x).view(x.size(0), -1))
