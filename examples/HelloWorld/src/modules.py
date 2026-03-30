from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torch

from dengine.models.decorators import register_model
from dengine.models.utils import get_unique_targets
from dengine.training_strategies.decorators import register_local_training
from dengine.models import ModuleBase
from dengine.training_strategies.local_update_strategy import TrainingEngine, training_step_output


@register_model()
class CustomNet(ModuleBase):
    def __init__(self, dataset: Subset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, len(get_unique_targets(dataset)))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


@register_local_training()
class CustomTrainingEngine(TrainingEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_func = nn.CrossEntropyLoss()

    def training_step(self, net: nn.Module, optimizer: Optimizer, images: Tensor, labels: Tensor) -> training_step_output:
        net.train()
        device = next(net.parameters()).device
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        erm = self._loss_func(output, labels)
        loss = torch.nanmean(erm, dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return training_step_output(output=output, reduced_loss=float(loss))
