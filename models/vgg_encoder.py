
import torch
import torch.nn as nn
from torchvision import models

class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(0,2)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(2,7)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(7,12)])
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(12,21)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        x = self.slice2(x)
        x = self.slice3(x)
        x = self.slice4(x)
        return x
