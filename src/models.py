"""Baseline CNN trained from scratch and ResNet-18 for transfer learning."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """Four conv blocks (32-64-128-256), global average pooling, dropout, linear classifier."""

    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def build_resnet18(pretrained=True, n_classes=2):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_model(name, cfg):
    if name == "baseline_cnn":
        return BaselineCNN(in_channels=cfg["channels"])
    if name == "resnet18":
        return build_resnet18(pretrained=cfg.get("pretrained", True))
    raise ValueError(name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
