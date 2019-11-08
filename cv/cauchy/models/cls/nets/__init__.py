from .alexnet import alexnet
from .resnet import gen_resnet
from .vgg import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)
from .squeezenet import squeezenet1_0, squeezenet1_1
from .inception import inception_v3
from .densenet import gen_densenet
from .vgg import gen_vggnet
from .squeezenet import gen_squeezenet
from .shufflenetv2 import gen_shufflenetv2

__all__ = [
    "alexnet",
    "gen_resnet",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "squeezenet1_0",
    "squeezenet1_1",
    "inception_v3",
    "gen_densenet",
    "gen_vggnet",
    "gen_squeezenet",
    "gen_shufflenetv2",
]
