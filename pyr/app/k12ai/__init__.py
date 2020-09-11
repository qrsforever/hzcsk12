#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file __init__.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-08-06 15:52

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence # noqa
from abc import ABC, abstractmethod
from collections import OrderedDict # noqa
from distutils.version import LooseVersion
from urllib.request import urlretrieve
from pprint import pprint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers.base import LightningLoggerBase # noqa
import os
import json
import sys, logging  # noqa
import warnings
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import GPUtil

from PIL import Image
from torch import optim
from torch.nn import functional as F
from torch.utils.data import (Dataset, DataLoader)

### ML
from sklearn.model_selection import train_test_split
from sklearn.base import (ClassifierMixin, RegressorMixin)
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix # noqa
from .ml.datasets import ML_load_dataset

warnings.filterwarnings('ignore')
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# np.set_printoptions(suppress=True)


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
    def step_lr(optimizer, step_size=30, gamma=0.1, last_epoch=-1):
        return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    @staticmethod
    def multistep_lr(optimizer, milestones, gamma=0.1, last_epoch=-1):
        return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    @staticmethod
    def reduceon_lr(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6):
        return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, verbose=False,
                mode=mode, factor=factor, patience=patience, min_lr=min_lr)


class EasyaiDataset(ABC, Dataset):
    def __init__(self, **kwargs):
        self._mean, self._std = None, None
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

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        self._std = std

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

    def set_img_trans(self, input_size:Union[Tuple[int, int], int, None], normalize=False):
        trans = []
        if input_size:
            if isinstance(input_size, int):
                input_size = (input_size, input_size)
            trans.append(Resize(input_size))
        trans.append(ToTensor())
        if normalize is True and self.mean and self.std:
            trans.append(Normalize(mean=self.mean, std=self.std))
        elif isinstance(normalize, list) and len(normalize) == 2:
            trans.append(Normalize(mean=normalize[0], std=normalize[1]))
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
            'rflowers': 22,    # 224 x 224
            'rflowers5': 5,    # 224 x 224
            'rchestxray': 2,   # 224 x 224
            'rfruits': 5       # 224 x 224
    }

    def __init__(self):
        super().__init__()
        self.datasets = {'train': None, 'val': None, 'test': None, 'predict': None}
        self.dataset_info = None

    def setup(self, stage: str):
        torch.cuda.empty_cache()

    def teardown(self, stage: str):
        for idx, gpu in enumerate(GPUtil.getGPUs()):
            allocmem = round(torch.cuda.memory_allocated(idx) / 1024**2, 2)
            allocmax = round(torch.cuda.max_memory_allocated(idx) / 1024**2, 2)
            print(f'({stage})\tGPU-{idx} memory allocated: {allocmem} MB\t max memory allocated: {allocmax} MB')
            # print(gpu)
        torch.cuda.empty_cache()

    def load_presetting_dataset_(self, dataset_name, dataset_root='/datasets/cv'):
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

        root = os.path.join(dataset_root, dataset_name)
        info = os.path.join(root, f'info.json')
        mean, std = None, None
        if os.path.exists(info):
            with open(info, 'r') as f:
                self.dataset_info = json.load(f)
                print('-' * 80)
                pprint(self.dataset_info)
                mean, std = self.dataset_info['mean'], self.dataset_info['std']
        datasets = {}
        for phase in ('train', 'val', 'test'):
            datasets[phase] = JsonfileDataset(path=root, phase=phase)
            datasets[phase].mean = mean
            datasets[phase].std = std
        return datasets

    def load_mnist(self):
        return self.load_presetting_dataset_('rmnist')

    def load_cifar10(self):
        return self.load_presetting_dataset_('rcifar10')

    def load_dogcat(self):
        return self.load_presetting_dataset_('rDogsVsCats')

    def load_flowers(self):
        return self.load_presetting_dataset_('rflowers')

    def load_flowers5(self):
        return self.load_presetting_dataset_('rflowers5')

    def load_fruits(self):
        return self.load_presetting_dataset_('rfruits')

    def load_chestxray(self):
        return self.load_presetting_dataset_('rchestxray')

    def load_houseprice(self):
        return ML_load_dataset('houseprice', dataset_root='/datasets/ml/house-prices')

    def load_sfcrime(self):
        return ML_load_dataset('sfcrime', dataset_root='/datasets/ml/sf-crime')

    def load_titanic(self):
        return ML_load_dataset('titanic', dataset_root='/datasets/ml/titanic')

    def prepare_dataset(self) -> Union[EasyaiDataset, List[EasyaiDataset], Dict[str, EasyaiDataset]]:
        return self.load_rmnist()

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

    def load_resnet18(self, num_classes, pretrained=True):
        return self.load_pretrained_model_('resnet18', num_classes=num_classes, pretrained=pretrained)

    def load_densenet121(self, num_classes, pretrained=True):
        return self.load_pretrained_model_('densenet121', num_classes=num_classes, pretrained=pretrained)

    def load_squeezenet10(self, num_classes, pretrained=True):
        return self.load_pretrained_model_('squeezenet1_0', num_classes=num_classes, pretrained=pretrained)

    def load_squeezenet11(self, num_classes, pretrained=True):
        return self.load_pretrained_model_('squeezenet1_1', num_classes=num_classes, pretrained=pretrained)

    def build_model(self):
        num_classes = self.dataset_info['num_classes'] if self.dataset_info else 1000
        return self.load_resnet18(num_classes)

    def configure_optimizer(self, model):
        # default
        return self.adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            base_lr=0.001)

    def configure_scheduler(self, optimizer):
        # default
        return self.step_lr(optimizer, step_size=30, gamma=0.1)

    def configure_optimizers(self):
        # TODO
        optimizers = self.configure_optimizer(self.model)
        schedulers = self.configure_scheduler(optimizers)
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        return optimizers, schedulers

    def configure_estimator(self) -> dict:
        return {'test_size': 0.4}

    def fit_end(self, preds, trues):
        print('fit result', accuracy_score(preds, trues))

    def predict_end(self, preds):
        print('predict result: ', preds)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def calculate_acc_(self, y_pred, y_true):
        return (torch.argmax(y_pred, dim=1) == y_true).float().mean()

    def step_(self, batch):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.cross_entropy(y_hat, y, reduction='mean')
        return x, y, y_hat, loss

    def get_dataloader(self, phase, batch_size=32, input_size=32,
                       data_augment=None, random_order=False,
                       normalize=False, num_workers=1,
                       drop_last=False, shuffle=False):
        if phase not in self.datasets.keys():
            raise RuntimeError(f'get {phase} data loader  error.')
        self.example_input_array = torch.zeros(batch_size, 3, input_size, input_size)
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
        return self.get_dataloader('train',
            batch_size=32,
            input_size=128,
            data_augment=[
                self.random_resized_crop(size=(128, 128)),
                self.random_brightness(factor=0.3),
                self.random_rotation(degrees=30)
            ],
            random_order=False,
            normalize=True,
            num_workers=1,
            drop_last=False,
            shuffle=False)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y, y_hat, loss = self.step_(batch)
        # with torch.no_grad():
        acc = self.calculate_acc_(y_hat, y)
        log = {'loss': loss, 'acc': acc}
        return log

    def training_epoch_end(self, outputs):
        log = {
            'train_loss': torch.stack([x['loss'] for x in outputs]).mean()
        }
        if 'acc' in outputs[0]:
            log['train_acc'] = torch.stack([x['acc'] for x in outputs]).mean()
        return {'progress_bar': log}

    ## Valid
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val',
                batch_size=32,
                input_size=128,
                num_workers=1,
                drop_last=False,
                shuffle=False)

    def validation_step(self, batch, batch_idx):
        x, y, y_hat, loss = self.step_(batch)
        acc = self.calculate_acc_(y_hat, y)
        log = {'val_loss': loss, 'val_acc': acc}
        return log

    def validation_epoch_end(self, outputs):
        log = {}
        if 'val_loss' in outputs[0]:
            log['val_loss'] = torch.stack([x['val_loss'] for x in outputs]).mean()
        if 'val_acc' in outputs[0]:
            log['val_acc'] = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'progress_bar': log}

    ## Test
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test',
                batch_size=32,
                input_size=128,
                num_workers=1,
                drop_last=False,
                shuffle=False)

    def test_step(self, batch, batch_idx):
        x, y, y_hat, loss = self.step_(batch)
        accuracy = self.calculate_acc_(y_hat, y)
        log = {'loss': loss, 'acc': accuracy}
        return log

    def test_epoch_end(self, outputs):
        log = {}
        if 'acc' in outputs[0]:
            log['test_acc'] = torch.stack([x['acc'] for x in outputs]).mean()
        return log

    def summarize(self, mode):
        print('\n' + '-' * 80)
        model_summary = ModelSummary(self, mode=mode)
        print('\n' + str(model_summary))
        return model_summary


class EasyaiRegressor(EasyaiClassifier):
    pass


class SSD300(nn.Module):
    # TODO

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.feature_provider = backbone
        self.num_classes = num_classes


class MultiBoxLoss(nn.Module):
    # TODO

    def __init__(self, overlap_thresh=0.5, neg_pos=3):
        super(MultiBoxLoss, self).__init__()
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos


class EasyaiDetector(EasyaiClassifier,
        IDataTransforms, ICriterion, IOptimizer, IScheduler):

    def __init__(self):
        super(EasyaiDetector, self).__init__()
        self.feature_extractor = SSD300()
        self.criterion = MultiBoxLoss()

    def forward(self, x):
        locs, confs = self.feature_extractor(x)
        return locs, confs

    def prepare_data(self):
        self.train_data = None # TODO VOCDataset()
        self.valid_data = None # TODO VOCDataset()

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=32,
            num_workers=1,
            shuffle=True,
            collate_fn=None, # TODO
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=32,
            num_workers=1,
            shuffle=False,
            collate_fn=None, # TODO
        )
        return valid_loader

    def training_step(self, batch, batch_idx):
        images, bboxes, bbox_labels = batch
        locs, confs = self(images)

        # TODO
        alpha = 0.5
        loc_loss, conf_loss = self.criterion(locs, confs, bboxes, bbox_labels)
        loss = conf_loss + alpha * loc_loss

        prog_dict = {'conf_l': conf_loss, 'loc_l': loc_loss}
        log_dict = {'conf_l': conf_loss, 'loc_l': loc_loss, 'train_loss': loss}
        return {'loss': loss, 'log': log_dict, 'progress_bar': prog_dict}

    def validation_step(self, batch, batch_idx):
        images, bboxes, bbox_labels = batch
        locs, confs = self(images)

        # TODO
        alpha = 0.5
        loc_loss, conf_loss = self.criterion(locs, confs, bboxes, bbox_labels)
        loss = conf_loss + alpha * loc_loss

        for i in range(locs.size(0)):
            # TODO
            pass

        confs = F.softmax(confs, dim=2)
        scores, idxs = confs.max(dim=2)

        # TODO
        return {'loc_loss': loc_loss, 'conf_loss': conf_loss, 'loss': loss, 'predictions' : '', 'gt' : (bboxes, bbox_labels)}

    def validation_epoch_end(self, outputs):
        conf_loss = sum([x['conf_loss'] for x in outputs])
        loc_loss = sum([x['loc_loss'] for x in outputs])
        loss = sum([x['loss'] for x in outputs])

        conf_loss /= len(outputs)
        loc_loss /= len(outputs)
        loss /= len(outputs)

        pred_boxes = []
        pred_scores = []
        pred_labels = []
        gt_boxes = []
        gt_labels = []

        for x in outputs:
            pred_boxes.extend(x['predictions'][0])
            pred_scores.extend(x['predictions'][1])
            pred_labels.extend(x['predictions'][2])

            gt_boxes.extend(x['gt'][0])
            gt_labels.extend(x['gt'][1])

        gt_labels = [x + 1 for x in gt_labels]

        # TODO mAP

        return {'val_loss': loss, 'log': {'val_loss': loss}, 'progress_bar': {'avg_val_loss': loss}}


class EasyaiTrainer(pl.Trainer):
    version = LooseVersion(pl.__version__).version
    model_summary = None
    log_lr = False

    def __init__(self,
            framework='cv',
            resume: bool = False,
            max_epochs: int = 10,
            max_steps: Optional[int] = None, # the times of iterations
            log_lr: bool = False,
            log_gpu_memory: Optional[str] = None, # 'min_max', 'all'
            model_summary: Optional[str] = None, # 'full', 'top'
            model_ckpt: Optional[dict] = None, # {'monitor': 'val_loss', 'period': 2, 'mode': 'min'}
            early_stop: Optional[dict] = None, # {'monitor': 'val_loss', 'patience': 3, 'mode': 'min'}
            log_rate: int = 1,
            ckpt_path: str = 'best' # 调试使用
            ): # noqa

        self.ml_model = None
        self.framework = framework
        if framework == 'cv':
            # class PrintLogger(LightningLoggerBase):
            #     def __init__(self):
            #         super().__init__()
            #
            #     @property
            #     def experiment(self):
            #         pass
            #
            #     @property
            #     def name(self):
            #         return 'easyaiecho'
            #
            #     @property
            #     def version(self):
            #         return '1.0'
            #
            #     def save(self):
            #         pass
            #
            #     def log_hyperparams(self, params):
            #         pass
            #
            #     def log_metrics(self, metrics, step):
            #         # print('-' * 80)
            #         # pprint(metrics)
            #         pass

            callbacks = []
            # if log_lr:
            #     from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
            #     lr_logger = LearningRateLogger(logging_interval='epoch')
            #     callbacks.append(lr_logger)
            self.log_lr = log_lr

            self.best_ckpt_path = f'/cache/checkpoints/{ckpt_path}.ckpt'
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
                cb_early_stop = EarlyStopping(**early_stop, verbose=True)

            self.model_summary = model_summary

            super(EasyaiTrainer, self).__init__(max_epochs=max_epochs, max_steps=max_steps,
                    logger=False, # PrintLogger(),
                    callbacks=callbacks,
                    progress_bar_refresh_rate=log_rate,
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
        if self.framework == 'cv':
            return super().fit(model=model)
        # ML
        self.framework = 'ml'
        self.ml_model = None
        ml_model = model.build_model()
        if not (isinstance(model, EasyaiClassifier)
                and isinstance(ml_model, ClassifierMixin)) \
                        and not (isinstance(model, EasyaiRegressor)
                                and isinstance(ml_model, RegressorMixin)):
            raise RuntimeError('model is invalid!')

        X, Y, feature_names, target_names = model.prepare_dataset()
        print(feature_names.tolist())
        params = model.configure_estimator()
        test_size = 0.4
        scaler = None
        if 'test_size' in params:
            test_size = params['test_size']
        if 'scaler_mode' in params:
            if 'zscore' == params['scaler_mode']:
                scaler = preprocessing.StandardScaler()
            elif 'minmax' == params['scaler_mode']:
                scaler = preprocessing.MinMaxScaler()

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_size)
        if scaler:
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)
        ml_model.fit(train_x, train_y)
        predict_y = ml_model.predict(test_x)
        model.fit_end(predict_y, test_y)
        # print('fit accuracy:', accuracy_score(predict_y, test_y))
        self.ml_model = ml_model

    def test(self, model=None):
        if self.framework == 'cv':
            if not os.path.exists(self.best_ckpt_path):
                raise FileNotFoundError('not found model weights file')
            save_oldval = self.resume_from_checkpoint
            self.resume_from_checkpoint = self.best_ckpt_path
            results = super().test(model=model, ckpt_path=self.best_ckpt_path, verbose=False)
            self.resume_from_checkpoint = save_oldval
            if results is not None and isinstance(results, list):
                print('-' * 80)
                for res in results:
                    pprint(res)
                    print('-' * 80)

    def predict(self, model, image_path=None, features=None, input_size=128):
        if self.framework == 'cv' and image_path is None:
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
                        if model.dataset_info and classidx < model.dataset_info['num_classes']:
                            target = model.dataset_info['label_names'][classidx]
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
        elif self.framework == 'ml':
            if self.ml_model is None or features is None:
                raise RuntimeError('ml model not run fit or lack features')
            if 1 == len(features.shape):
                features = np.expand_dims(features, 0)
            result = self.ml_model.predict(features)
            model.predict_end(result)
            # print('predict result:', result)

    def on_validation_start(self):
        if self.progress_bar_callback:
            self.progress_bar_callback.disable()

        if self.log_lr:
            lrs = []
            for scheduler in self.lr_schedulers:
                ss = scheduler['scheduler']
                if isinstance(ss, optim.lr_scheduler.ReduceLROnPlateau):
                    for i, param_group in enumerate(ss.optimizer.param_groups):
                        lrs.append(np.float32(param_group['lr']))
                else:
                    lrs.extend([np.float32(x) for x in ss.get_lr()])
            self.add_progress_bar_metrics({'lr': lrs})
        return super().on_validation_start()

    def on_validation_end(self):
        if self.progress_bar_callback:
            self.progress_bar_callback.enable()
        return super().on_validation_end()

    def on_fit_start(self, model):
        return super().on_fit_start(model)

    def on_fit_end(self):
        return super().on_fit_end()

    def on_test_start(self):
        return super().on_test_start()

    def on_test_end(self):
        return super().on_test_end()

    def save_checkpoint(self, filepath, weights_only: bool = False):
        return super().save_checkpoint(self.best_ckpt_path, weights_only)

# High-Performance Computing system (HPC)
