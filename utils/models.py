import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


class Classifier(nn.Module):
    def __init__(self,
                 num_classes=10,
                 pretrained=True,
                 input_channels=36):

        nn.Module.__init__(self)
        self.classifier = resnet34(pretrained=pretrained)


        # replace fc layer and first convolutional layer
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    def forward(self, x):
        # print(x.shape)
        pred = self.classifier.forward(x)
        return pred
