import torch
from torch import nn
import torch.nn.functional as F

from dengine.dataset import SupervisedDataset

from .decorators import register_model
from .decorators import ModuleBase as dModule


# To calculate the number of parameters in a convolutional layers:
# ((kernel_width * kernel_height * in_channels)+1)* out_channels
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        # h1 = max([100, int(dim_in/2)])
        # h2 = max([50, int(h1/2)])
        h1 = 512
        h2 = 256
        h3 = 128
        self.fc1 = nn.Linear(dim_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, dim_out)
        # self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(self.relu(x))
        return x


@register_model()
class CNNMnist(dModule):
    def __init__(
        self,
        dataset: SupervisedDataset,
        *args,
        num_channels: int,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, len(dataset.unique_targets))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class MnistNet(nn.Module):
    def __init__(self, args):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x


class CNNFashion(nn.Module):
    def __init__(self, args):
        super(CNNFashion, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7, stride=1)
        )
        self.fc = nn.Linear(256, args.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
        return x


class CNNEMnist(nn.Module):
    def __init__(self, args, number_of_classes):
        super(CNNEMnist, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=128),
            nn.Dropout(p=0.5),
            nn.Linear(128, number_of_classes)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        # x = torch.flatten(x, 1)
        # print(x.shape)
        # x = self.fc(x)
        return x


class EMNISTCNN(nn.Module):
    def __init__(self, args, fmaps1, fmaps2, dense, dropout):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=args.num_channels, out_channels=fmaps1, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=fmaps1, out_channels=fmaps2, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fcon1 = nn.Sequential(nn.Linear(49 * fmaps2, dense), nn.LeakyReLU())
        self.fcon2 = nn.Linear(dense, args.num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x
