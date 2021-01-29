#!/bin/python3

import os
import torch
import torchvision
import torch.optim as optim
from time import time
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from k12ai.common.log_message import MessageMetric
from k12ai.mnist.network import Discriminator, Generator

torch.backends.cudnn.enabled = True
torch.manual_seed(8888)


class MnistGAN():
    def __init__(self, latent_dim, dataloader,
                 batch_size=32, device='cpu', lr_d=1e-3, lr_g=2e-4):

        def make_noise_points(num):
            return torch.randn((num, latent_dim), device=device)

        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        self.noise_fn = make_noise_points
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.total_iters = 0
        self.device = device
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(self.discriminator.parameters(),
                                  lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.999))
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

    def generate_samples(self, latent_vec=None, num=None):
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()
        return samples

    def train_step_generator(self):
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self, real_samples):
        self.discriminator.zero_grad()

        # real samples
        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        # combine
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_epoch(self, mm, checkpoints_dir, print_freq=100, save_latest_freq=512):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch, (real_samples, _) in enumerate(self.dataloader):
            self.total_iters += self.batch_size
            real_samples = real_samples.to(self.device)
            ldr_, ldf_ = self.train_step_discriminator(real_samples)
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_
            loss_g_running += self.train_step_generator()
            if self.total_iters % print_freq == 0:
                G_loss = round(loss_g_running / (batch + 1), 3)
                Dr_loss = round(loss_d_real_running / (batch + 1), 3)
                Df_loss = round(loss_d_fake_running / (batch + 1), 3)
                print(f"{batch+1}/{len(self.dataloader)}:"
                      f" G={G_loss},"
                      f" Dr={Dr_loss},"
                      f" Df={Df_loss}",
                      end='\r',
                      flush=True)
                mm.add_scalar('训练', 'G_loss', x=self.total_iters, y=G_loss)
                mm.add_scalar('训练', 'Dr_loss', x=self.total_iters, y=Dr_loss)
                mm.add_scalar('训练', 'Df_loss', x=self.total_iters, y=Df_loss).send()
            if self.total_iters % save_latest_freq == 0:
                torch.save(self.generator, f'{checkpoints_dir}/G.pth') 
                torch.save(self.discriminator, f'{checkpoints_dir}/D.pth') 

        if print_freq:
            print()
        loss_g_running /= batch
        loss_d_real_running /= batch
        loss_d_fake_running /= batch
        return (loss_g_running, (loss_d_real_running, loss_d_fake_running))


def train_mnist_gan(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))])

    dataset = ImageFolder(
            root=opt.dataroot,
            transform=transform)

    dataloader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            drop_last=True,
            num_workers=opt.num_threads)

    G_W_PATH = f'{opt.checkpoints_dir}/G.pth'
    D_W_PATH = f'{opt.checkpoints_dir}/D.pth'

    gan = MnistGAN(opt.latent_dim, dataloader, device=device, batch_size=opt.batch_size, lr_g=opt.lr)
    if opt.continue_train:
        if os.path.exists(G_W_PATH):
            gan.generator.load_state_dict(torch.load(G_W_PATH))
        if os.path.exists(D_W_PATH):
            gan.discriminator.load_state_dict(torch.load(D_W_PATH))
    start = time()
    mm = MessageMetric()
    os.makedirs(opt.checkpoints_dir, exist_ok=True)
    num_epochs = 2 * opt.n_epochs
    for i in range(1, num_epochs + 1):
        print(f"Epoch {i}; Elapsed time = {int(time() - start)}s")
        gan.train_epoch(mm, opt.checkpoints_dir, opt.print_freq, opt.save_latest_freq)
        mm.add_scalar('训练', '进度', x=i, y=round(i / num_epochs, 2)).send()

    torch.save(gan.generator.state_dict(), G_W_PATH) 
    torch.save(gan.discriminator.state_dict(), D_W_PATH) 
