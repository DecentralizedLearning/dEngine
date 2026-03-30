import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.dropout_rate = dropout_rate

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.shortcut = nn.Identity()
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)

        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        nclasses: int,
        input_channels: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'Depth must be 6n+4'
        n = (depth - 4) // 6

        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(
            input_channels,
            nStages[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.layer1 = self._make_layer(nStages[0], nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(nStages[1], nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(nStages[2], nStages[3], n, stride=2, dropout_rate=dropout_rate)

        self.bn = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(nStages[3], nclasses)
        # self._initialize()

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout_rate):
        layers = [WideBasicBlock(in_planes, out_planes, stride, dropout_rate)]
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_planes, out_planes, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
