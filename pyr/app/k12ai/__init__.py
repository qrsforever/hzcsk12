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
from distutils.version import LooseVersion
from urllib.request import urlretrieve

import os
import json
import warnings
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn

from PIL import Image
from torch import optim
from torch.nn import functional as F
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


class ICriterion(object):
    @staticmethod
    def cross_entropy(input, target, reduction='mean', ignore_index=-100):
        # return nn.CrossEntropyLoss(reduction=reduction)
        return F.cross_entropy(input, target, reduction=reduction, ignore_index=ignore_index)


class IOptimizer(object):
    @staticmethod
    def adam(params, base_lr, betas=(0.9, 0.999), weight_decay=0, amsgrad=False):
        return optim.Adam(
                params, lr=base_lr, betas=betas,
                weight_decay=weight_decay, amsgrad=amsgrad)

    @staticmethod
    def sgd(params, base_lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        return optim.SGD(
                params, lr=base_lr, momentum=0, dampening=0,
                weight_decay=0, nesterov=False)


class IScheduler(object):
    @staticmethod
    def period_step(optimizer, step_size=30, gamma=0.1, last_epoch=-1):
        return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    @staticmethod
    def multi_step(optimizer, milestones, gamma=0.1, last_epoch=-1):
        return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones, gamma=gamma, last_epoch=last_epoch)


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
        if transforms and len(transforms) > 0:
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


class EasyaiClassifier(pl.LightningModule,
        IDataTransforms, ICriterion, IOptimizer, IScheduler):
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
    DATASETS = {
            'rmnist': 10,      # 28 x 28
            'rcifar10': 10,    # 32 x 32
            'rDogsVsCats': 2,  # 224 x 224
            'flowers': 22,     # 224 x 224
            'chestxray': 2     # 224 x 224
    }

    def __init__(self):
        super(EasyaiClassifier, self).__init__()
        self.dataset_classes = None
        self.datasets = {'train': None, 'val': None, 'test': None, 'predict': None}

    def setup(self, stage: str):
        pass

    def teardown(self, stage: str):
        pass

    # Data
    def load_presetting_dataset_(self, dataset_name, dataset_root='/datasets'):
        if dataset_name not in self.DATASETS.keys():
            raise ValueError(f'{dataset_name} is not in {self.DATASETS.keys()}')

        class JsonfileDataset(EasyaiDataset):
            def data_reader(self, path, phase):
                """
                Args:
                    path: the dataset root directory
                    phase: the json file name (train.json / val.json / test.json)
                """
                image_list = []
                label_list = []

                with open(os.path.join(path, f'{phase}.json'), 'r') as f:
                    items = json.load(f)
                    for item in items:
                        image_list.append(os.path.join(path, item['image_path']))
                        label_list.append(item['label'])
                return image_list, label_list

        mean, std = None, None
        root = os.path.join(dataset_root, dataset_name)
        info = os.path.join(root, f'info.json')
        if os.path.exists(info):
            with open(info, 'r') as f:
                ii = json.load(f)
                mean, std = ii['mean'], ii['std']
                self.dataset_classes = ii['classes']
        datasets = {
            'train': JsonfileDataset(mean=mean, std=std, path=root, phase='train'),
            'val': JsonfileDataset(mean=mean, std=std, path=root, phase='val'),
            'test': JsonfileDataset(mean=mean, std=std, path=root, phase='test'),
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

    def prepare_data(self): # private
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

        self.model = self.build_model()

    # Model
    def load_pretrained_model_(self, model_name, num_classes, pretrained=True):
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
        num_classes = 1000
        if self.dataset_classes:
            num_classes = len(self.dataset_classes)
        return self.load_pretrained_model_('resnet18', num_classes=num_classes)

    def configure_optimizer(self, model):
        # default
        return self.adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            base_lr=0.001)

    def configure_scheduler(self, optimizer):
        # default
        return self.period_step(optimizer, step_size=30, gamma=0.1)

    def configure_optimizers(self):
        # TODO
        optimizer = self.configure_optimizer(self.model)
        scheduler = self.configure_scheduler(optimizer)
        return [optimizer], [scheduler]

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def calculate_acc_(self, y_pred, y_true):
        return (torch.argmax(y_pred, axis=1) == y_true).float().mean()

    def step_(self, batch):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.cross_entropy(y_hat, y, reduction='mean')
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
        # REQUIRED
        x, y, y_hat, loss = self.step_(batch)
        with torch.no_grad():
            accuracy = self.calculate_acc_(y_hat, y)
        log = {'train_loss': loss, 'train_acc': accuracy}
        output = OrderedDict({
            'loss': loss,
            **log
        })
        return output

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        log = {'train_loss': avg_loss, 'train_acc': avg_acc}
        return {**log}

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
        log = {'val_loss': loss, 'val_acc': accuracy}
        return log

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        log = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'progress_bar': log}

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
        log = {'test_loss': loss, 'test_acc': accuracy}
        output = OrderedDict({
            'progress_bar': log,
            **log
        })
        return output

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        log = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'progress_bar': log}


class EasyaiTrainer(pl.Trainer):
    version = LooseVersion(pl.__version__).version
    best_ckpt_path = '/cache/checkpoints/best.ckpt'

    def __init__(self,
            resume: bool = False,
            max_epochs: int = 10,
            max_steps: Optional[int] = None, # the times of iterations
            log_gpu_memory: Optional[str] = None, # 'min_max', 'all'
            model_summary: str = 'top', # 'full', 'top'
            model_ckpt: Optional[dict] = None, # {'monitor': 'val_loss', 'period': 2, 'mode': 'min'}
            early_stop: Optional[dict] = None, # {'monitor': 'val_loss', 'patience': 3, 'mode': 'min'}
            log_save_interval: int = 100, progress_bar_rate: int = 10
            ): # noqa

        resume_from_checkpoint = None
        if resume and os.path.exists(self.best_ckpt_path):
            resume_from_checkpoint = self.best_ckpt_path

        cb_model_ckpt = True
        if model_ckpt:
            if not isinstance(model_ckpt, dict):
                raise TypeError('model_ckpt must be dict type!')
            validkeys = ('monitor', 'period', 'mode')
            if any([it not in validkeys for it in model_ckpt.keys()]):
                raise KeyError(f'{model_ckpt.keys()} is not in {validkeys}')
            from pytorch_lightning.callbacks import ModelCheckpoint
            cb_model_ckpt = ModelCheckpoint(**model_ckpt)

        cb_early_stop = False
        if early_stop:
            if not isinstance(early_stop, dict):
                raise TypeError('early_stop must be dict type!')
            validkeys = ('monitor', 'patience', 'mode')
            if any([it not in validkeys for it in early_stop.keys()]):
                raise KeyError(f'{early_stop.keys()} is not in {validkeys}')
            from pytorch_lightning.callbacks import EarlyStopping
            cb_early_stop = EarlyStopping(**early_stop)

        super(EasyaiTrainer, self).__init__(max_epochs=max_epochs, max_steps=max_steps,
                logger=False,
                log_save_interval=log_save_interval, progress_bar_refresh_rate=progress_bar_rate,
                log_gpu_memory=log_gpu_memory, weights_summary=model_summary,
                num_sanity_val_steps=0,
                checkpoint_callback=cb_model_ckpt,
                early_stop_callback=cb_early_stop,
                default_root_dir='/cache',
                resume_from_checkpoint=resume_from_checkpoint,
                check_val_every_n_epoch=1,
                val_check_interval=1.0,
                gpus=[0])

    def fit(self, model):
        return super().fit(model=model)

    def test(self, model=None):
        if self.version == [0, 8, 5]:
            return super().test(model=model)
        else:
            return super().test(model=model, verbose=False)

    def predict(self, model, image_path, input_size=128):
        class ImageFolderDataset(EasyaiDataset):
            def data_reader(self, path):
                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                image_list = []
                label_list = []
                for filename in os.listdir(path):
                    if not filename.lower().endswith(extensions):
                        continue
                    image_list.append(f'{path}/{filename}')
                    label_list.append(-1)
                return (image_list, label_list)

        def predict_dataloader(self):
            if image_path.startswith('oss://'):
                dataset = ImageFolderDataset(path=os.path.join('/cache', image_path[6:]))
            elif image_path.startswith('http') or image_path.startswith('ftp'):
                dstpath = '/cache/user_images'
                if not os.path.exists(dstpath):
                    os.makedirs(dstpath)
                urlretrieve(image_path, os.path.join(dstpath, os.path.basename(image_path)))
                dataset = ImageFolderDataset(path=dstpath)
            else:
                raise NotImplementedError('not in (oss, http, ftp)')
            dataset.set_img_trans(input_size=input_size)
            dataloader = DataLoader(dataset, num_workers=1, shuffle=False, batch_size=32, drop_last=False)
            return dataloader

        def predict_step(self, batch, batch_idx):
            x, _, p = batch
            y_hat = self(x)
            return list(zip(p, F.softmax(y_hat, dim=1).cpu().numpy().astype(float)))

        def predict_epoch_end(self, outputs):
            result = {}
            for item in outputs:
                for path, props in item:
                    filename = os.path.basename(path)
                    classidx = np.argmax(props)
                    if model.dataset_classes and classidx < len(model.dataset_classes):
                        target = model.dataset_classes[classidx]
                    else:
                        target = classidx
                    result[filename] = target
            return result

        try:
            m_test_dataloader = getattr(model.__class__, 'test_dataloader', None)
            m_test_step = getattr(model.__class__, 'test_step', None)
            m_test_epoch_end = getattr(model.__class__, 'test_epoch_end', None)
            setattr(model.__class__, 'test_dataloader', predict_dataloader)
            setattr(model.__class__, 'test_step', predict_step)
            setattr(model.__class__, 'test_epoch_end', predict_epoch_end)
            if self.version == [0, 8, 5]:
                return super().test(model=model)
            else:
                return super().test(model=model, verbose=False)
        except Exception as err:
            raise err
        finally:
            if m_test_dataloader:
                setattr(model.__class__, 'test_dataloader', m_test_dataloader)
            if m_test_step:
                setattr(model.__class__, 'test_step', m_test_step)
            if m_test_epoch_end:
                setattr(model.__class__, 'test_epoch_end', m_test_epoch_end)

    def on_fit_start(self):
        return super().on_fit_start()

    def on_fit_end(self):
        return super().on_fit_end()

    def on_test_start(self):
        return super().on_test_start()

    def on_test_end(self):
        return super().on_test_end()

    def save_checkpoint(self, filepath, weights_only: bool = False):
        return super().save_checkpoint(self.best_ckpt_path, weights_only)

# High-Performance Computing system (HPC)
