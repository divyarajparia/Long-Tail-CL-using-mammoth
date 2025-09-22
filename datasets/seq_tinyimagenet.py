# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from backbone.ResNetBlock import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils import smart_joint
from utils.conf import base_path
from datasets.utils import set_default_from_args


class TinyImagenet(Dataset):
    """Defines the Tiny Imagenet dataset."""

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        # self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.not_aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # if download:
        #     if os.path.isdir(root) and len(os.listdir(root)) > 0:
        #         print('Download not needed, files already on disk.')
        #     else:
        #         from onedrivedownloader import download

        #         print('Downloading dataset')
        #         ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
        #         download(ln, filename=smart_joint(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)

        # self.data = [] # this should have the paths of all images read from txt file
        # for num in range(20):
        #     self.data.append(np.load(smart_joint(
        #         root, 'processed/x_%s_%02d.npy' %
        #               ('train' if self.train else 'val', num + 1))))
        # self.data = np.concatenate(np.array(self.data))

        # self.targets = []
        # for num in range(20):
        #     self.targets.append(np.load(smart_joint(
        #         root, 'processed/y_%s_%02d.npy' %
        #               ('train' if self.train else 'val', num + 1))))
        # self.targets = np.concatenate(np.array(self.targets))
        # import os

        # Determine which txt file to use (train or test)
        txt_file = os.path.join(root, 'train.txt') if self.train else os.path.join(root, 'test.txt')

        self.data = []     # list to store image paths
        self.targets = []  # list to store targets

        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line should be like: 
                # "imagenet_resized_256/train/n01729322/n01729322_4311.JPEG 0"
                parts = line.split()
                # Use the first part as the image path and the second as the target label.
                # Optionally, you can convert the image path into an absolute path if needed:
                img_path = os.path.join(root, parts[0])
                self.data.append(img_path)
                self.targets.append(int(parts[1]))

    def __len__(self):
        return len(self.data)

    # # Cant do this
    # def __getitem__(self, index):
    #     img, target = self.data[index], self.targets[index]

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(np.uint8(255 * img))
    #     original_img = img.copy()

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     if hasattr(self, 'logits'):
    #         return img, target, original_img, self.logits[index]

    #     return img, target
    def __getitem__(self, index):
        # Get the file path and target from lists
        img_path, target = self.data[index], self.targets[index]
        img_path = str(img_path)
        # Open the image using PIL
        img = Image.open(img_path).convert("RGB")
        
        # Make a copy of the original image
        original_img = img.copy()
        
        # Apply any specified transforms
        if self.transform is not None:
            img = self.transform(img)
            # img = img.clone()
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]
        
        return img, target

class IMBALANCETinyImagenet(TinyImagenet):
    cls_num = 100  

    def __init__(self, root: str, imb_type='exp', imb_factor=0.1, rand_number=0,
                 train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(IMBALANCETinyImagenet, self).__init__(root, train, transform, target_transform, download)

        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'fewshot':
            for cls_idx in range(cls_num):
                if cls_idx < cls_num // 2:
                    num = img_max
                else:
                    num = img_max * 0.01
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.extend([self.data[i] for i in selec_idx])
            new_targets.extend([the_class] * the_img_num)

        # Instead of using np.vstack, simply assign the list back.
        self.data = new_data
        self.targets = new_targets
    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]

    # def __getitem__(self, index):
    #     img, target = self.data[index], self.targets[index]

    #     img = Image.fromarray(np.uint8(255 * img))
    #     original_img = img.copy()

    #     not_aug_img = self.not_aug_transform(original_img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     if hasattr(self, 'logits'):
    #         return img, target, not_aug_img, self.logits[index]

    #     return img, target, not_aug_img
    def __getitem__(self, index):
        # Get file path and target
        img_path, target = self.data[index], self.targets[index]
        img_path = str(img_path)
        
        # Open the image file using PIL and convert to RGB
        img = Image.open(img_path).convert("RGB")
        
        # Make a copy of the original image for the not-augmented version
        original_img = img.copy()
        
        # Create the not-augmented image using the not_aug_transform (e.g., ToTensor)
        # not_aug_img = self.not_aug_transform(original_img)
        not_aug_img = self.transform(original_img)

        
        # Apply additional transforms if specified
        if self.transform is not None:
            img = self.transform(img)
            # # Clone to ensure the returned tensor has resizable storage.
            # img = img.clone().contiguous()
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # If there are extra outputs (like logits), handle them similarly.
        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        
        return img, target, not_aug_img
class MyTinyImagenet(TinyImagenet):
    """Overrides the TinyImagenet dataset to change the getitem function."""

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    # def __getitem__(self, index):
    #     img, target = self.data[index], self.targets[index]

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(np.uint8(255 * img))
    #     original_img = img.copy()

    #     not_aug_img = self.not_aug_transform(original_img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     if hasattr(self, 'logits'):
    #         return img, target, not_aug_img, self.logits[index]

    #     return img, target, not_aug_img
    def __getitem__(self, index):
        # Get file path and target
        img_path, target = self.data[index], self.targets[index]
        img_path = str(img_path)
        
        # Open the image file using PIL and convert to RGB
        img = Image.open(img_path).convert("RGB")
        
        # Make a copy of the original image for the not-augmented version
        original_img = img.copy()
        
        # Create the not-augmented image using the not_aug_transform (e.g., ToTensor)
        not_aug_img = self.not_aug_transform(original_img)
        
        # Apply additional transforms if specified
        if self.transform is not None:
            img = self.transform(img)
            # Clone to ensure the returned tensor has resizable storage.
            # img = img.clone().contiguous()
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # If there are extra outputs (like logits), handle them similarly.
        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        
        return img, target, not_aug_img

class SequentialTinyImagenet(ContinualDataset):
    """The Sequential Tiny Imagenet dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    SIZE = (64, 64)
    TRANSFORM = transforms.Compose(
        ### First resize to 256 and crop to 224 instead of 64 Made the 64 to 224 here
        [transforms.Resize(256),
         transforms.RandomCrop(224, padding=4),
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
        transform = self.TRANSFORM

        # test_transform = trandsor(), self.get_normalization_transform()])

        test_transform = transforms.Compose([
        transforms.Resize(256),       # Resize so the smaller edge is 256
        transforms.CenterCrop(224),   # Crop the central 224×224 region
        transforms.ToTensor(),
        self.get_normalization_transform()
    ])
        # train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
        #                                train=True, download=True, transform=transform)
        base_path = "/data1/es22btech11013/DATA/imagenet"
        # train_dataset = IMBALANCETinyImagenet(base_path() + 'TINYIMG',
        #                                train=True, download=True, transform=transform)
        
        # test_dataset = TinyImagenet(base_path() + 'TINYIMG',
        #                             train=False, download=True, transform=test_transform)


        ### This is where the loaders are called - so this is where the resizing needs to be done
        # --- START OF FINAL, CORRECT MODIFICATION ---
        if hasattr(self.args, 'long_tail') and self.args.long_tail:
            print(f"INFO: Creating LONG-TAIL TinyImageNet with imbalance factor: {self.args.imb_factor}")
            train_dataset = IMBALANCETinyImagenet(
                base_path,
                train=True,
                download=True,
                transform=transform,
                imb_factor=self.args.imb_factor
            )
        else:
            print("INFO: Creating BALANCED TinyImageNet.")
            train_dataset = IMBALANCETinyImagenet(
                base_path,
                train=True,
                download=True,
                transform=transform
            )

        test_dataset = TinyImagenet(
            base_path,
            train=False,
            download=True,
            transform=test_transform
        )
        # --- END OF FINAL, CORRECT MODIFICATION ---

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"
    
    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialTinyImagenet.MEAN, SequentialTinyImagenet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialTinyImagenet.MEAN, SequentialTinyImagenet.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(CLASS_NAMES, self.args)
        self.class_names = classes
        return self.class_names


CLASS_NAMES = ['hognose_snake', 'cock', 'wardrobe', 'corkscrew', 'isopod', 'beaver', 'acorn', 'goldfinch', 'Siamese_cat', 'chiffonier', 'bittern', 'screw', 'cairn', 'valley', 'lens_cap', 'Brittany_spaniel', 'Appenzeller', 'entertainment_center', 'Greater_Swiss_Mountain_dog', 'Band_Aid', 'dhole', 'sea_anemone', 'ice_cream', 'thresher', 'chime', 'sunglass', 'can_opener', 'microphone', 'quail', 'Brabancon_griffon', 'computer_keyboard', 'hand-held_computer', 'eel', 'Norwegian_elkhound', 'mailbox', 'leopard', 'mitten', 'cocker_spaniel', 'worm_fence', 'dowitcher', 'tennis_ball', 'Afghan_hound', 'parking_meter', 'snow_leopard', 'spiny_lobster', 'monarch', 'hook', 'drumstick', 'toilet_tissue', 'lumbermill', 'coho', 'remote_control', 'chain_mail', 'swimming_trunks', 'white_stork', 'teddy', 'moped', 'buckeye', 'holster', 'ping-pong_ball', 'purse', 'indigo_bunting', 'wolf_spider', 'beacon', 'sturgeon', 'toaster', 'Arctic_fox', 'doormat', 'black_widow', 'bullet_train', 'vending_machine', 'cricket', 'long-horned_beetle', 'rock_python', 'red_wine', 'assault_rifle', 'carbonara', 'screen', 'confectionery', 'academic_gown', 'cannon', 'loudspeaker', 'African_hunting_dog', 'plow', 'koala', 'crutch', 'groenendael', 'Norwich_terrier', 'carton', 'combination_lock', 'candle', 'Windsor_tie', 'panpipe', 'hip', 'cabbage_butterfly', 'space_shuttle', 'chow', 'wool', 'binder', 'alligator_lizard']