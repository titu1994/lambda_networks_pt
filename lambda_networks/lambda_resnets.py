# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from lambda_networks import lambda_module_2d as lambda_module


__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_lambda_conv: bool = False,
        lambda_k: int = 16,
        lambda_m: Optional[int] = None,
        lambda_r: Optional[int] = None,
        lambda_u: int = 1,
        lambda_heads: int = 4,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.use_lambda_conv = use_lambda_conv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        if use_lambda_conv:
            self.conv2 = lambda_module.LambdaLayer2D(
                planes, planes, dim_k=lambda_k, m=lambda_m, r=lambda_r, heads=lambda_heads, dim_intra=lambda_u
            )
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_lambda_conv: bool = False,
        lambda_k: int = 16,
        lambda_m: Optional[int] = None,
        lambda_r: Optional[int] = None,
        lambda_u: int = 1,
        lambda_heads: int = 4,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.use_lambda_conv = use_lambda_conv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        if use_lambda_conv:
            ops = [
                lambda_module.LambdaLayer2D(
                    planes,
                    planes,
                    dim_k=lambda_k,
                    m=lambda_m,
                    r=lambda_r,
                    heads=lambda_heads,
                    dim_intra=lambda_u,
                )
            ]

            if downsample is not None:
                ops.append(nn.AvgPool2d(kernel_size=3, stride=stride, padding=1))

            self.conv2 = nn.Sequential(*ops)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # Incorporate ResNet-B/C/D from https://arxiv.org/abs/1812.01187.
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = True,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        lambda_blocks: Optional[List[int]] = None,
        lambda_k: int = 16,
        lambda_m: bool = False,
        lambda_r: Optional[int] = None,
        lambda_u: int = 1,
        lambda_heads: int = 4,
        input_size: Optional[int] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        if lambda_m and input_size is None:
            raise RuntimeError("If `lambda_m` has been provided, then input image size (input_size) must be provided !")

        if lambda_m and lambda_r is not None:
            raise ValueError("Either set global context (lambda_m=True) or local context (lambda_r=R}")

        self.input_size = input_size

        if lambda_m is False:
            lambda_m = None

        # ResNet-D
        self.conv1_1 = nn.Conv2d(3, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_pool_input = nn.AvgPool2d(2, 2)

        self.conv1_2 = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_4 = nn.Conv2d(32, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # divide image size by 8x after 4x pool in initial branch + 2x stride in second block
        if input_size is not None:
            input_size = input_size // 8

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            lambda_blocks=lambda_blocks,
            lambda_k=lambda_k,
            lambda_m=lambda_m,
            lambda_r=lambda_r,
            lambda_u=lambda_u,
            lambda_heads=lambda_heads,
            input_size=input_size,
        )

        # Half input size from previous stride
        if input_size is not None:
            input_size = input_size // 2

        # divide input size by 2
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            lambda_blocks=list(range(layers[3])),
            lambda_k=lambda_k,
            lambda_m=lambda_m,
            lambda_r=lambda_r,
            lambda_u=lambda_u,
            lambda_heads=lambda_heads,
            input_size=input_size,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        lambda_blocks: Optional[List[int]] = None,
        lambda_k: int = 16,
        lambda_m: Optional[int] = None,
        lambda_r: Optional[int] = None,
        lambda_u: int = 1,
        lambda_heads: int = 4,
        input_size: Optional[int] = None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = [
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
            ]
            if stride > 1:
                downsample.append(nn.AvgPool2d(3, 2, padding=1))
            downsample.append(norm_layer(planes * block.expansion))

            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for layer_idx in range(1, blocks):
            if lambda_blocks is not None and layer_idx in lambda_blocks:
                lambda_kwargs = dict(
                    use_lambda_conv=True,
                    lambda_k=lambda_k,
                    lambda_m=input_size // 2 if (lambda_m and input_size is not None) else None,
                    lambda_r=lambda_r,
                    lambda_u=lambda_u,
                    lambda_heads=lambda_heads,
                )
            else:
                lambda_kwargs = dict(use_lambda_conv=False)

            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    **lambda_kwargs,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x_1 = self.conv1_1(x)
        x_1 = self.avg_pool_input(x_1)

        # ResNet-D path
        x_2 = self.conv1_2(x)
        x_2 = self.conv1_3(x_2)
        x_2 = self.conv1_4(x_2)

        x = x_1 + x_2

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], **kwargs: Any) -> ResNet:
    print(f"Building model LambdaResnet-{arch}")
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(
    lambda_k: int = 16,
    lambda_m: bool = False,
    lambda_r: Optional[int] = None,
    lambda_u: int = 1,
    lambda_heads: int = 4,
    input_size: Optional[int] = None,
    **kwargs: Any,
) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(
        'resnet18',
        BasicBlock,
        [2, 2, 2, 2],
        lambda_blocks=[1],  # 0 based index
        lambda_k=lambda_k,
        lambda_m=lambda_m,
        lambda_r=lambda_r,
        lambda_u=lambda_u,
        lambda_heads=lambda_heads,
        input_size=input_size,
        **kwargs,
    )


def resnet34(
    lambda_k: int = 16,
    lambda_m: bool = False,
    lambda_r: Optional[int] = None,
    lambda_u: int = 1,
    lambda_heads: int = 4,
    input_size: Optional[int] = None,
    **kwargs: Any,
) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(
        'resnet34',
        BasicBlock,
        [3, 4, 6, 3],
        lambda_blocks=[2],  # 0 based index
        lambda_k=lambda_k,
        lambda_m=lambda_m,
        lambda_r=lambda_r,
        lambda_u=lambda_u,
        lambda_heads=lambda_heads,
        input_size=input_size,
        **kwargs,
    )


def resnet50(
    lambda_k: int = 16,
    lambda_m: bool = False,
    lambda_r: Optional[int] = None,
    lambda_u: int = 1,
    lambda_heads: int = 4,
    input_size: Optional[int] = None,
    **kwargs: Any,
) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(
        'resnet50',
        Bottleneck,
        [3, 4, 6, 3],
        lambda_blocks=[2],  # 0 based index
        lambda_k=lambda_k,
        lambda_m=lambda_m,
        lambda_r=lambda_r,
        lambda_u=lambda_u,
        lambda_heads=lambda_heads,
        input_size=input_size,
        **kwargs,
    )


def resnet101(
    lambda_k: int = 16,
    lambda_m: bool = False,
    lambda_r: Optional[int] = None,
    lambda_u: int = 1,
    lambda_heads: int = 4,
    input_size: Optional[int] = None,
    **kwargs: Any,
) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(
        'resnet101',
        Bottleneck,
        [3, 4, 23, 3],
        lambda_blocks=[5, 11, 17],  # 0 based index
        lambda_k=lambda_k,
        lambda_m=lambda_m,
        lambda_r=lambda_r,
        lambda_u=lambda_u,
        lambda_heads=lambda_heads,
        input_size=input_size,
        **kwargs,
    )


def resnet152(
    lambda_k: int = 16,
    lambda_m: bool = False,
    lambda_r: Optional[int] = None,
    lambda_u: int = 1,
    lambda_heads: int = 4,
    input_size: Optional[int] = None,
    **kwargs: Any,
) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(
        'resnet152',
        Bottleneck,
        [3, 8, 36, 3],
        lambda_blocks=[4, 9, 14, 19, 24, 29],  # 0 based index
        lambda_k=lambda_k,
        lambda_m=lambda_m,
        lambda_r=lambda_r,
        lambda_u=lambda_u,
        lambda_heads=lambda_heads,
        input_size=input_size,
        **kwargs,
    )
