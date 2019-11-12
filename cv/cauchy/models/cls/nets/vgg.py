import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from cauchy.utils.tools.logger import Logger as Log
from cauchy.utils.tools.load_custom_model import gen_model_urls

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vgg11(pretrained=False, model_urls=None, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["A"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_11"]))
        Log.info("finished loading pretrained weights for vgg11")
    return model


def vgg11_bn(pretrained=False, model_urls=None, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["A"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_11_bn"]))
        Log.info("finished loading pretrained weights for vgg11_bn")
    return model


def vgg13(pretrained=False, model_urls=None, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["B"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_13"]))
        Log.info("finished loading pretrained weights for vgg13")
    return model


def vgg13_bn(pretrained=False, model_urls=None, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["B"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_13_bn"]))
        Log.info("finished loading pretrained weights for vgg13_bn")
    return model


def vgg16(pretrained=False, model_urls=None, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["D"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_16"]))
        Log.info("finished loading pretrained weights for vgg16")
    return model


def vgg16_bn(pretrained=False, model_urls=None, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["D"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_16_bn"]))
        Log.info("finished loading pretrained weights for vgg16_bn")
    return model


def vgg19(pretrained=False, model_urls=None, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["E"]), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_19"]))
        Log.info("finished loading pretrained weights for vgg19")
    return model


def vgg19_bn(pretrained=False, model_urls=None, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg["E"], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg_19_bn"]))
        Log.info("finished loading pretrained weights for vgg19_bn")
    return model


def gen_vggnet(model_name=None, pretrained=False, weights_host=None, **kwargs):
    model_files = {
        "vgg_11": "vgg11-bbd30ac9.pth",
        "vgg_13": "vgg13-c768596a.pth",
        "vgg_16": "vgg16-397923af.pth",
        "vgg_19": "vgg19-dcbb9e9d.pth",
        "vgg_11_bn": "vgg11_bn-6002323d.pth",
        "vgg_13_bn": "vgg13_bn-abd245e5.pth",
        "vgg_16_bn": "vgg16_bn-6c64b313.pth",
        "vgg_19_bn": "vgg19_bn-c79401a0.pth",
    }
    model_urls = None
    if pretrained:
        model_urls = gen_model_urls(
            weights_host=weights_host, model_files=model_files
        )
    if model_name == "vgg_11":
        model = vgg11(pretrained=pretrained, model_urls=model_urls, **kwargs)
    elif model_name == "vgg_13":
        model = vgg13(pretrained=pretrained, model_urls=model_urls, **kwargs)
    elif model_name == "vgg_16":
        model = vgg16(pretrained=pretrained, model_urls=model_urls, **kwargs)
    elif model_name == "vgg_19":
        model = vgg19(pretrained=pretrained, model_urls=model_urls, **kwargs)
    elif model_name == "vgg_11_bn":
        model = vgg11_bn(pretrained=pretrained, model_urls=model_urls, **kwargs)
    elif model_name == "vgg_13_bn":
        model = vgg13_bn(pretrained=pretrained, model_urls=model_urls, **kwargs)
    elif model_name == "vgg_16_bn":
        model = vgg16_bn(pretrained=pretrained, model_urls=model_urls, **kwargs)
    elif model_name == "vgg_19_bn":
        model = vgg19_bn(pretrained=pretrained, model_urls=model_urls, **kwargs)
    else:
        raise NotImplementedError
    return model
