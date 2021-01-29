#!/bin/python3

import torch
import torchvision

from k12ai.mnist.network import Generator
from k12ai.common.log_message import MessageMetric

torch.manual_seed(8888)


def predict_mnist_gan(opt):
    mm = MessageMetric()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(opt.latent_dim).to(device)
    G.load_state_dict(torch.load(f'{opt.checkpoints_dir}/G.pth'))
    G.eval()
    with torch.no_grad():
        latent_vec = torch.randn((25, opt.latent_dim), device=device)
        samples = G(latent_vec).cpu() * -1
    ims = torchvision.utils.make_grid(samples, nrow=5, normalize=True)
    ims = ims.numpy().transpose((1, 2, 0))
    mm.add_image('测试', '5x5', ims).send()
