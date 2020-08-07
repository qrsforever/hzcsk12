#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file __init__.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-08-06 15:52

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence # noqa
from abc import ABC, abstractmethod
from collections import OrderedDict

import os
import json
import warnings
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn

from torch import optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import (Dataset, DataLoader)

warnings.filterwarnings('ignore')

from torchvision.transforms import ( # noqa
        Resize,
        Compose,
        ToTensor,
        Normalize,
        RandomOrder,
        ColorJitter,
        RandomRotation,
        RandomGrayscale,
        RandomResizedCrop,
        RandomVerticalFlip,
        RandomHorizontalFlip)


class IDataTransforms(object):

    @staticmethod
    def compose_order(transforms: List):
        return Compose(transforms)

    @staticmethod
    def shuffle_order(transforms: List):
        return RandomOrder(transforms)

    @staticmethod
    def image_resize(size):
        return Resize(size=size)

    @staticmethod
    def image_to_tensor():
        return ToTensor()

    @staticmethod
    def normalize_tensor(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        return Normalize(mean, std)

    @staticmethod
    def random_brightness(factor=0.25):
        return ColorJitter(brightness=factor)

    @staticmethod
    def random_contrast(factor=0.25):
        return ColorJitter(contrast=factor)

    @staticmethod
    def random_saturation(factor=0.25):
        return ColorJitter(saturation=factor)

    @staticmethod
    def random_hue(factor=0.25):
        return ColorJitter(hue=factor)

    @staticmethod
    def random_rotation(degrees=25):
        return RandomRotation(degrees=degrees)

    @staticmethod
    def random_grayscale(p=0.5):
        return RandomGrayscale(p=p)

    @staticmethod
    def random_resized_crop(size):
        return RandomResizedCrop(size=size)

    @staticmethod
    def random_horizontal_flip(p=0.5):
        return RandomHorizontalFlip(p=p)

    @staticmethod
    def random_vertical_flip(p=0.5):
        return RandomVerticalFlip(p=p)


class EasyaiDataset(ABC, Dataset):
    def __init__(self, mean=None, std=None, **kwargs):
        self.mean, self.std = mean, std
        self.augtrans = None
        self.imgtrans = ToTensor()

        # reader
        data = self.data_reader(**kwargs)
        if isinstance(data, (tuple, list)) and len(data) == 2:
            self.images, self.labels = data
        elif isinstance(data, dict) and all([x in data.keys() for x in ('images', 'labels')]):
            self.images, self.labels = data['images'], data['labels']
        else:
            raise ValueError('The return of data_reader must be List or Dict')

    @abstractmethod
    def data_reader(self, **kwargs) -> Union[Tuple[List, List, List], Dict[str, List]]:
        """
        (M)
        """

    def set_aug_trans(self, transforms:Union[list, None], random_order=False):
        if transforms:
            if any([not hasattr(x, '__call__') for x in transforms]):
                raise ValueError(f'set_aug_trans: transforms params is invalid.')
            if random_order:
                self.augtrans = RandomOrder(transforms)
            else:
                self.augtrans = Compose(transforms)

    def set_img_trans(self, input_size:Union[Tuple[int, int], int, None], normalize=True):
        trans = []
        if input_size:
            trans.append(Resize(input_size))
        trans.append(ToTensor())
        if normalize and self.mean and self.std:
            trans.append(Normalize(mean=self.mean, std=self.std))
        self.imgtrans = Compose(trans)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.augtrans:
            img = self.augtrans(img)
        return self.imgtrans(img), self.labels[index], self.images[index]

    def __len__(self):
        return len(self.images)


class EasyaiClassifier(pl.LightningModule, IDataTransforms):
    BACKBONES = [
        'resnet18',
        'resnet50',
        'densenet121',
        'mobilenet_v2',
        'squeezenet1_0',
        'squeezenet1_1',
        'shufflenet_v2_x0_5',
        'shufflenet_v2_x1_0',
    ]

    def __init__(self):
        super(EasyaiClassifier, self).__init__()
        self.model = self.build_model()
        self.criterion = None
        self.datasets = {'train': None, 'val': None, 'test': None}

    def setup(self, stage: str):
        pass

    def teardown(self, stage: str):
        pass

    # Data
    def load_presetting_dataset_(self, dataset_name, dataset_root='/datasets'):
        class JsonfileDataset(EasyaiDataset):
            def data_reader(self, path, phase):
                """
                Args:
                    path: the dataset root directory
                    phase: the json file name (train.json / val.json / test.json)
                """
                image_list = []
                label_list = []
                with open(os.path.join(path, f'{phase}.json')) as f:
                    items = json.load(f)
                    for item in items:
                        image_list.append(os.path.join(path, item['image_path']))
                        label_list.append(item['label'])
                return image_list, label_list

        root = os.path.join(dataset_root, dataset_name)
        datasets = {
            'train': JsonfileDataset(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), path=root, phase='train'),
            'val': JsonfileDataset(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), path=root, phase='val'),
            'test': JsonfileDataset(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), path=root, phase='test'),
        }
        return datasets

    def prepare_dataset(self) -> Union[EasyaiDataset, List[EasyaiDataset], Dict[str, EasyaiDataset]]:
        return self.load_presetting_dataset_('rmnist')

    @staticmethod
    def _safe_delattr(cls, method):
        try:
            delattr(cls, method)
        except Exception:
            pass

    def prepare_data(self):
        datasets = self.prepare_dataset()
        if isinstance(datasets, EasyaiDataset):
            self._safe_delattr(self.__class__, 'val_dataloader')
            self._safe_delattr(self.__class__, 'validation_step')
            self._safe_delattr(self.__class__, 'validation_epoch_end')
            self._safe_delattr(self.__class__, 'test_dataloader')
            self._safe_delattr(self.__class__, 'test_step')
            self._safe_delattr(self.__class__, 'test_epoch_end')
            self.datasets['train'] = datasets
        elif isinstance(datasets, (list, tuple)) and len(datasets) <= 3:
            self.datasets['train'] = datasets[0]
            self.datasets['val'] = datasets[1]
            if len(datasets) == 2:
                self._safe_delattr(self.__class__, 'test_dataloader')
                self._safe_delattr(self.__class__, 'test_step')
                self._safe_delattr(self.__class__, 'test_epoch_end')
            else:
                self.datasets['test'] = datasets[2]

        elif isinstance(datasets, dict) and \
                all([x in datasets.keys() for x in ('train', 'val', 'test')]):
            self.datasets = datasets
        else:
            raise ValueError('the return of prepare_dataset is invalid.')

    # Model
    def load_pretrained_model_(self, model_name, num_classes:int = 1000, pretrained=True):
        if model_name not in self.BACKBONES:
            raise ValueError(f'{model_name} is not in {self.BACKBONES}')
        model = getattr(torchvision.models, model_name)(pretrained)
        if num_classes != 1000:
            if model_name.startswith('vgg'):
                model.classifier[6] = nn.Linear(4096, num_classes)
            elif model_name.startswith('resnet'):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name.startswith('alexnet'):
                model.classifier[6] = nn.Linear(4096, num_classes)
            elif model_name.startswith('mobilenet_v2'):
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_name.startswith('squeezenet'):
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
            elif model_name.startswith('shufflenet'):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name.startswith('densenet'):
                in_features = {
                    "densenet121": 1024,
                    "densenet161": 2208,
                    "densenet169": 1664,
                    "densenet201": 1920,
                }
                model.classifier = nn.Linear(in_features[model_name], num_classes)
            else:
                raise NotImplementedError(f'{model_name}')
        return model

    def build_model(self):
        return self.load_pretrained_model_('resnet18', 10)

    # Hypes
    @property
    def loss(self):
        if self.criterion is None:
            self.criterion = self.configure_criterion()
        return self.criterion

    def configure_criterion(self):
        # default
        loss = nn.CrossEntropyLoss(reduction='mean')
        return loss

    def configure_optimizer(self):
        # default
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001)
        return optimizer

    def configure_scheduler(self, optimizer):
        # default
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return scheduler

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        scheduler = self.configure_scheduler(optimizer)
        return [optimizer], [scheduler]

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def calculate_acc_(self, y_pred, y_true):
        return (torch.argmax(y_pred, axis=1) == y_true).float().mean()

    def step_(self, batch):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return x, y, y_hat, loss

    def get_dataloader(self, phase,
                       data_augment=None, random_order=False,
                       normalize=False, input_size=None,
                       batch_size=32, num_workers=1,
                       drop_last=False, shuffle=False):
        if phase not in self.datasets.keys():
            raise RuntimeError(f'get {phase} data loader  error.')
        dataset = self.datasets[phase]
        dataset.set_aug_trans(transforms=data_augment, random_order=random_order)
        dataset.set_img_trans(input_size=input_size, normalize=normalize)
        return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=drop_last,
                shuffle=shuffle)

    ## Train
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            phase='train',
            data_augment=[
                self.random_resized_crop(size=(128, 128)),
                self.random_brightness(factor=0.3),
                self.random_rotation(degrees=30)
            ],
            random_order=False,
            input_size=128,
            normalize=True,
            batch_size=32,
            num_workers=1,
            drop_last=False,
            shuffle=False)

    def training_step(self, batch, batch_idx):
        x, y, y_hat, loss = self.step_(batch)
        with torch.no_grad():
            accuracy = self.calculate_acc_(y_hat, y)
        log = {'train_loss': loss, 'train_acc': accuracy}
        output = OrderedDict({
            'loss': loss,        # M
            'acc': accuracy,     # O
            'progress_bar': log, # O
            # 'log': log           # O
        })
        return output

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        log = {'train_loss': avg_loss, 'train_acc': avg_acc}
        output = OrderedDict({
            'progress_loss': log,
            # 'log': log
        })
        return output

    ## Valid
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val',
                input_size=128,
                batch_size=32,
                num_workers=2,
                drop_last=False,
                shuffle=False)

    def validation_step(self, batch, batch_idx):
        x, y, y_hat, loss = self.step_(batch)
        accuracy = self.calculate_acc_(y_hat, y)
        output = OrderedDict({
            'val_loss': loss,
            'val_acc': accuracy,
        })
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        log = {'val_loss': avg_loss, 'val_acc': avg_acc}
        output = OrderedDict({
            'progress_loss': log,
            # 'log': log
        })
        return output

    ## Test
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test',
                input_size=128,
                batch_size=32,
                num_workers=1,
                drop_last=False,
                shuffle=False)

    def test_step(self, batch, batch_idx):
        x, y, y_hat, loss = self.step_(batch)
        accuracy = self.calculate_acc_(y_hat, y)
        output = OrderedDict({
            'test_loss': loss,
            'test_acc': accuracy
        })
        return output

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        log = {'test_loss': avg_loss, 'test_acc': avg_acc}
        output = OrderedDict({
            'progress_bar': log,
            # 'log': log
        })
        return output


class EasyaiTrainer(pl.Trainer):
    version = pl.__version__

    def __init__(self, max_epochs: int = 30, max_steps: Optional[int] = None,
            log_save_interval: int = 100, progress_bar_rate: int = 10,
            log_gpu_memory: Optional[str] = None, model_summary: str = 'top', # 'full', 'top'
            ): # noqa

        super(EasyaiTrainer, self).__init__(max_epochs=max_epochs, max_steps=max_steps,
                logger=False,
                log_save_interval=log_save_interval, progress_bar_refresh_rate=progress_bar_rate,
                log_gpu_memory=log_gpu_memory, weights_summary=model_summary,
                num_sanity_val_steps=0,
                default_root_dir='/cache', gpus=[0])

    def fit(self, model):
        return super().fit(model=model)

    def test(self, model=None):
        if self.version.startswith('0.8'):
            return super().test(model=model)
        else:
            return super().test(model=model, verbose=False)

    def on_fit_end(self):
        return super().on_fit_end()

    def on_test_start(self):
        return super().on_test_start()

    def on_test_end(self):
        return super().on_test_end()
