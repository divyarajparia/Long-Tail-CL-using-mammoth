# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100
import numpy as np

from backbone.ResNetBlock import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class IMBALANCECIFAR100(CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        print(f"\n=== IMBALANCING CIFAR-100 CLASSES ===")
        print(f"Imbalance type: {imb_type}")
        print(f"Imbalance factor: {imb_factor}")
        print(f"Total classes: {cls_num}")

        img_max = len(self.data) / cls_num
        print(f"Original samples per class: {img_max}")

        img_num_per_cls = []
        if imb_type == 'exp':
            print(f"Applying exponential imbalancing...")
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))

            print(f"After exponential imbalancing:")
            print(f"  Max class (class 0): {img_num_per_cls[0]} samples")
            print(f"  Min class (class {cls_num-1}): {img_num_per_cls[-1]} samples")
            print(f"  Imbalance ratio: {img_num_per_cls[0] / img_num_per_cls[-1]:.2f}")

        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'fewshot':
            for cls_idx in range(cls_num):
                if cls_idx<50:
                    num = img_max
                else:
                    num = img_max*0.01
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img



class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):
    """Sequential CIFAR100 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data."""

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    def __init__(self, args: Namespace, imb_factor: float = 0.1) -> None:
        """Initialize SequentialCIFAR100 dataset.

        Args:
            args: Arguments namespace containing hyperparameters
            imb_factor: Imbalance factor for the dataset (default: 0.1)
        """
        super().__init__(args)
        self.imb_factor = imb_factor

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)), #Added this line for l2p, as it needs 224 * 224 images
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
        
# slca was gicing error so added this TEST_TRANSFORM
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        print(f"\n=== DATASET INITIALIZATION ===")
        print(f"Dataset: {self.NAME}")
        print(f"Setting: {self.SETTING}")
        print(f"N_TASKS: {self.N_TASKS}")
        print(f"N_CLASSES_PER_TASK: {self.N_CLASSES_PER_TASK}")
        print(f"Imbalance factor: {self.imb_factor}")
        print(f"Current task: {self.c_task + 1}")

        transform = self.TRANSFORM

        # test_transform = transforms.Compose(
        #     [transforms.ToTensor(), self.get_normalization_transform()])

        # This is for the l2p model, as it uses a ViT backbone
        test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        self.get_normalization_transform()
    ])


        # train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
        #                            download=True, transform=transform)
        # test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False,
        #                          download=True, transform=test_transform)

        print(f"Creating IMBALANCED CIFAR-100 dataset with imb_factor={self.imb_factor}")
        # Use the imb_factor from instance attribute
        train_dataset = IMBALANCECIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=transform, imb_factor=self.imb_factor)
        test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False,
                                 download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR100.MEAN, SequentialCIFAR100.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR100.MEAN, SequentialCIFAR100.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('lr_scheduler') #Commenting this because moe_adapter does not require lr_scheduler
    def get_scheduler_name(self):
        return 'multisteplr'

    @set_default_from_args('lr_milestones')
    def get_scheduler_name(self):
        return [35, 45]

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR100(base_path() + 'CIFAR100', train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
