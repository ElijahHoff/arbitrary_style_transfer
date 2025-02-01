
import torch
import torch.nn as nn

from .utils import adain
from .utils import calc_mean_std

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


