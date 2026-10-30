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


class UNetTrunk(nn.Module):
    """Спільний енкодер-декодер; голови (детермінована чи з невизначеністю) додаються окремо."""

    def __init__(self, in_ch, base=24):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

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
        return d1

    @staticmethod
    def _pad_to_match(x, ref):
        diff_h, diff_w = ref.shape[-2] - x.shape[-2], ref.shape[-1] - x.shape[-1]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [0, diff_w, 0, diff_h])
        return x


class ResidualUNet(nn.Module):
    """Крок 2: ŷ = y_ref + Δθ(x), лише детермінована поправка."""

    def __init__(self, in_ch, out_ch, base=24, zero_init=False):
        super().__init__()
        self.trunk = UNetTrunk(in_ch, base)
        self.out = nn.Conv2d(base, out_ch, 1)
        if zero_init:
            # Розд. 2.4: Δθ ≡ 0 на старті, тож модель починає рівно з рівня базового методу.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.trunk(x))  # residual Δ, без активації (може бути +/-)


class ResidualUNetUncertainty(nn.Module):
    """Метод 2 концепції: гетероскедастична голова - модель повертає (Δ, log σ).

    ŷ = y_ref + Δθ(x), σ = exp(log_σθ(x)) (clamp для стабільності навчання).
    Дві окремі 1x1-голови на тому самому стовбурі - дешевше за ансамбль/дифузію,
    це "основний варіант" з концепції, не порівняльний дифузійний.
    """

    def __init__(self, in_ch, out_ch, base=24, log_sigma_min=-7.0, log_sigma_max=3.0):
        super().__init__()
        self.trunk = UNetTrunk(in_ch, base)
        self.delta_head = nn.Conv2d(base, out_ch, 1)
        self.log_sigma_head = nn.Conv2d(base, out_ch, 1)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, x):
        feat = self.trunk(x)
        delta = self.delta_head(feat)
        log_sigma = self.log_sigma_head(feat).clamp(self.log_sigma_min, self.log_sigma_max)
        return delta, log_sigma
