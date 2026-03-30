from torch import nn
from torch import Tensor
import torchvision.models as models
from transformers import ViTModel, ViTConfig

from torch.utils.data import Subset
import torch.nn.functional as F

from .resnet import WideResNet
from .decorators import register_model, ModuleBase
from .modules import DepthWiseSkipConvolution, NormGeluDownScalingBlock
from .modules import PatchDiscriminator, Nonlinear2DClassificationHead
from .utils import get_image_channels, get_image_size, get_unique_targets


@register_model()
class PatchCNNClassifier(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        discriminator_encoder_spreadout_channels: int,
        discriminator_encoder_output_channels: int,
        discriminator_output_size: int,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)
        self._discriminator_output_features = (
            discriminator_output_size * discriminator_output_size
        )

        labels = get_unique_targets(dataset)
        dataset_channels = get_image_channels(dataset)
        dataset_size = get_image_size(dataset)

        self.discriminator = PatchDiscriminator(
            input_size=dataset_size,
            input_channels=dataset_channels,
            output_size=discriminator_output_size,
            encoder_spreadout_channels=discriminator_encoder_spreadout_channels,
            encoder_output_channels=discriminator_encoder_output_channels,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self._discriminator_output_features, len(labels)),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        features = self.discriminator(x)
        flatted_features = features.view(-1, self._discriminator_output_features)
        return self.output_layer(flatted_features)


@register_model()
class ResNetClassifier(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        labels = get_unique_targets(dataset)
        dataset_channels = get_image_channels(dataset)

        self.backbone = models.resnet18(weights=None)

        if dataset_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                dataset_channels, 64, kernel_size=3, stride=2, padding=3, bias=False
            )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, len(labels)),
        )

    def forward(self, x: Tensor):
        return self.backbone(x)


@register_model()
class MobileNetV3Classifier(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        labels = get_unique_targets(dataset)
        dataset_channels = get_image_channels(dataset)

        self.backbone = models.mobilenet_v3_small(weights=None)

        if dataset_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                dataset_channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, len(labels))

    def forward(self, x: Tensor):
        return self.backbone(x)


@register_model()
class SqueezeNetClassifier(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        labels = get_unique_targets(dataset)
        dataset_channels = get_image_channels(dataset)

        self.backbone = models.squeezenet1_1(weights=None)

        if dataset_channels != 3:
            self.backbone.features[0] = nn.Conv2d(
                dataset_channels,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        final_conv = nn.Conv2d(512, len(labels), kernel_size=1)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: Tensor):
        x = self.backbone(x)
        return x.view(x.size(0), -1)


@register_model()
class WideResNetClassifier(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        depth: int,
        widen_factor: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        labels = get_unique_targets(dataset)
        dataset_channels = get_image_channels(dataset)

        self.backbone = WideResNet(
            depth=depth,
            widen_factor=widen_factor,
            dropout_rate=dropout,
            nclasses=len(labels),
            input_channels=dataset_channels
        )

    def forward(self, x: Tensor):
        return self.backbone(x)


@register_model()
class TorchWideResNetClassifier(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        imagenet_pretrained: bool = False,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        labels = get_unique_targets(dataset)
        dataset_channels = get_image_channels(dataset)

        self.backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights if imagenet_pretrained else None
        )

        if dataset_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                dataset_channels, 64, kernel_size=3, stride=2, padding=3, bias=False
            )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(labels))
        )

    def forward(self, x: Tensor):
        return self.backbone(x)


@register_model()
class ViT(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        image_size: int = 32,
        num_channels: int = 3,
        dropout: float = 0.0,

        patch_size: int = 4,

        hidden_size: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,

        **kwargs
    ):
        super().__init__(dataset, *args, **kwargs)
        nclasses = len(get_unique_targets(dataset))

        config = ViTConfig.from_dict({
            "image_size": image_size,
            "patch_size": patch_size,
            "num_channels": num_channels,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "intermediate_size": mlp_dim,
            "hidden_act": "gelu",
            "hidden_dropout_prob": dropout,
            "attention_probs_dropout_prob": dropout,
            "layer_norm_eps": 1e-6,
            "qkv_bias": True,
        })

        self.vit = ViTModel(config)
        self.classifier = nn.Linear(hidden_size, nclasses)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        out = self.vit(pixel_values=x)
        cls_token = out.last_hidden_state[:, 0]
        return self.classifier(cls_token)


@register_model()
class TinyCNN(ModuleBase):
    """Inspired by the work of https://github.com/soyflourbread/cifar10-tiny/blob/main/nn.py#L193
    """
    def __init__(
        self,
        dataset: Subset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        nclasses = len(get_unique_targets(dataset))
        dataset_channels = get_image_channels(dataset)

        self._input_projection = nn.Sequential(
            nn.Conv2d(dataset_channels, out_channels=32, kernel_size=3, padding='same'),
            nn.GroupNorm(num_groups=1, num_channels=32, eps=1e-3, affine=True)
        )
        self._feature_extractor = nn.Sequential(
            DepthWiseSkipConvolution(32, 2),
            DepthWiseSkipConvolution(32, 2),
            NormGeluDownScalingBlock(32, 64),  # 32x32 -> 16x16
            *[DepthWiseSkipConvolution(64, 2) for _ in range(3)],
            NormGeluDownScalingBlock(64, 96),  # 16x16 -> 8x8
            *[DepthWiseSkipConvolution(96, 2) for _ in range(2)],
        )
        self._head = Nonlinear2DClassificationHead(in_channels=96, out_dimension=nclasses)

    def forward(self, x: Tensor):
        x = self._input_projection(x)
        x = self._feature_extractor(x)
        return self._head(x)


@register_model()
class BasicCNN(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        nclasses = len(get_unique_targets(dataset))
        dataset_channels = get_image_channels(dataset)

        self.conv1 = nn.Conv2d(dataset_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


@register_model()
class CNNMnist(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)
        nclasses = len(get_unique_targets(dataset))

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
