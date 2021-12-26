'''
ResNet, https://arxiv.org/abs/1512.03385
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,  planes, kernel_size=3, stride=stride, padding=1, bias=False)   # key convolution
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 64
        self.deconv_with_bias = False
        self.channels = [16, 32, 64, 128, 256, 512]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,  64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)

        # used for deconv layers
        self.deconv_layers_1 = self._make_deconv_layer(256, 4)
        self.deconv_layers_2 = self._make_deconv_layer(128, 4)
        self.deconv_layers_3 = self._make_deconv_layer(64, 4)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes,
                                                 planes * block.expansion,
                                                 kernel_size=1,
                                                 stride=stride,
                                                 bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    
    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_filters, num_kernels):
        layers = []
        kernel, padding, output_padding = self._get_deconv_cfg(num_kernels)
        planes = num_filters
        layers.append(
            nn.ConvTranspose2d(
                in_channels=self.in_planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        up_level_16 = self.deconv_layers_1(out)
        up_level_8 = self.deconv_layers_2(up_level_16)
        up_level_4 = self.deconv_layers_3(up_level_8)

        return [up_level_4, up_level_8, up_level_16]


    def init_deconv(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def init_weights(self, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            self.init_deconv(self.deconv_layers_1)
            self.init_deconv(self.deconv_layers_2)
            self.init_deconv(self.deconv_layers_3)


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        model.init_weights()
    return model


def resnet34(pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        model.init_weights()
    return model


def resnet50(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        model.init_weights()
    return model


def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        model.init_weights()
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        print('===> loading imagenet pretrained model.')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
        model.init_weights()
    return model



if __name__ == '__main__':
    import torch
    net = resnet50(pretrained=True)
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())