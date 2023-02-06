'''
This is the test code of poisoned training under BadNets.
'''

import os
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, dataloader
import numpy as np
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import torchvision.models as models
import core
import argparse



parser = argparse.ArgumentParser(description='PyTorch BadNets_ImageNet')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu_id
datasets_root_dir = '/Codes/pytorch-tiny-imagenet-master/tiny-imagenet-200'


# ========== ResNet-18_ImageNet_BadNets ==========
resnet18 = models.resnet18(pretrained=False)
resnet18.fc = torch.nn.Linear(512, 200)
my_model = resnet18

# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip(),
# ])

# trainset = DatasetFolder(root=os.path.join(datasets_root_dir, 'train'),
#                          transform=transform_train,
#                          loader=cv2.imread,
#                          extensions=('jpeg',),
#                          target_transform=None,
#                          is_valid_file=None,
#                          )

transform_test = Compose([
    ToTensor()
])

testset = DatasetFolder(root=os.path.join(datasets_root_dir, 'val'),
                         transform=transform_test,
                         loader=cv2.imread,
                         extensions=('jpeg',),
                         target_transform=None,
                         is_valid_file=None,
                         )


test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False)


test_samples = torch.zeros((10000, 3, 64, 64))

test_labels = torch.zeros((10000, 1))



for batch_id, batch in enumerate(test_loader):
    batch_img = batch[0]
    batch_label = batch[1]
    test_samples[batch_id, :, :, :] = batch_img
    test_labels[batch_id] = batch_label
    if (batch_id + 1) % 100 == 0:
        print((batch_id + 1)/100)

torch.save(test_samples, 'benign_test_samples.pth')
torch.save(test_labels, 'benign_labels.pth')

# pattern = torch.zeros((64, 64), dtype=torch.uint8)
# pattern[-8:, -8:] = torch.tensor(np.random.randint(low=0, high=256, size=64).reshape(8, 8))

# weight = torch.zeros((64, 64), dtype=torch.float32)
# weight[-8:, -8:] = 1.0


# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=my_model,
#     loss=nn.CrossEntropyLoss(),
#     y_target=0,
#     poisoned_rate=0.1,
#     pattern=pattern,
#     weight=weight,
#     seed=global_seed,
#     deterministic=deterministic
# )

# # Train Attacked Model
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 2,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 20,

#     'save_dir': 'experiments',
#     'experiment_name': 'ResNet-18_ImageNet_BadNets'
# }
# badnets.train(schedule)

# poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
# torch.save(poisoned_test_dataset, 'poisoned_test_dataset_BadNets.pth')

