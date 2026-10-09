"""Невеликий залишковий U-Net для кроку 2: ŷ = y_ref + Δθ(x).

Легкий навмисно (канали 24-48-96, 2 рівні зменшення роздільності) - тренується
на CPU (ноутбук без GPU) за розумний час на одному ROI. Для повного масштабу
(більше ROI, повна роздільність 256x256x13) знадобиться GPU - сюди лише
закладено архітектуру, масштабування - окреме рішення.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=24):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = self._pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)  # residual Δ, без активації на виході (може бути +/-)

    @staticmethod
    def _pad_to_match(x, ref):
        diff_h, diff_w = ref.shape[-2] - x.shape[-2], ref.shape[-1] - x.shape[-1]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [0, diff_w, 0, diff_h])
        return x
