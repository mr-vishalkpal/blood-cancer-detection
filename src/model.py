import torch.nn as nn
from torchvision import models


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, num_classes=4, pretrained=True):
    if name == "simple_cnn":
        return SimpleCNN(num_classes)

    if name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        return m

    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
        return m

    raise ValueError(f"Unknown model: {name}")