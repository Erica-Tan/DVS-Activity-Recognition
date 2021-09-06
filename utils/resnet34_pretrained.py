import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.resnet import resnet34


class ResNet34(nn.Module):
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
        pred = self.classifier.forward(x)
        return pred


def test():
    x = torch.randn((16, 36, 128, 128))
    model = ResNet34()
    preds = model(x)
    print(preds.shape)
    
    summary(model, input_size=(36, 128, 128), device='cpu')


if __name__ == "__main__":
    test()