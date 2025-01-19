
import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Пример упрощённого декодера, который 'разворачивает' фичи (512 каналов после relu4_1 в VGG19)
    обратно в 3-канальное изображение. Структура может быть скорректирована по вашим нуждам.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # По сути, мы делаем обратную операцию к той части VGG, которую вырезали.
        # Здесь слои могут отличаться, но этот вариант часто используют для AdaIN.
        self.decode = nn.Sequential(
            # Начинаем с 512 каналов
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.decode(x)

