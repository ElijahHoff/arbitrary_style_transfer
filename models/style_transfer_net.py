
import torch
import torch.nn as nn

from utils import adain
from utils import calc_mean_std

def calc_mean_std(feat, eps=1e-5):
    """
    Вычисляет среднее и стандартное отклонение по каждому каналу (C) в батче.
    feat: (N, C, H, W)
    Возвращает mean, std в формах (N, C, 1, 1).
    """
    N, C = feat.shape[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat, eps=1e-5):
    """
    Adaptive Instance Normalization:
      1) нормируем контент-фичи (вычесть mean, поделить на std)
      2) умножить на std стиля и добавить mean стиля
    """
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

class StyleTransferNet(nn.Module):
    """
    Объединяет Encoder и Decoder, выполняя перенос стиля через AdaIN.
    Применение:
      net = StyleTransferNet(encoder, decoder)
      out = net(content_img, style_img, alpha=1.0)
    """
    def __init__(self, encoder, decoder):
        super(StyleTransferNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, content, style, alpha=1.0):
        # 1) Получаем фичи контента и стиля
        content_feat = self.encoder(content)
        style_feat   = self.encoder(style)
        # 2) AdaIN для получения стилизованных фичей
        t = adain(content_feat, style_feat)
        # 3) Смешиваем стилизованные фичи с исходными (если alpha < 1.0)
        t = alpha * t + (1 - alpha) * content_feat
        # 4) Декодируем обратно в изображение
        out = self.decoder(t)
        return out


