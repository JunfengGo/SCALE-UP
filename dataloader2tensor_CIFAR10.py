'''
This is the code of obtaining samples from a given dataloader and save them as a tensor.
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
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu_id
dataloader_root_dir = 'poisoned_test_dataset_WaNet_VGG.pth'
#dataloader_root_dir = 'poisoned_test_dataset_ISSBA_VGG.pth'
#dataloader_root_dir = 'poisoned_test_dataset_BadNets_VGG.pth'
#dataloader_root_dir = 'poisoned_test_dataset_TUAP_VGG.pth'
#dataloader_root_dir = 'poisoned_test_dataset_PhysicalBA_VGG.pth'


poisoned_test_dataloader = torch.load(dataloader_root_dir)
poisoned_test_samples = torch.zeros((10000, 3, 32, 32))

for batch_id, batch in enumerate(poisoned_test_dataloader):
    batch_img = batch[0]
    batch_label = batch[1]
    poisoned_test_samples[batch_id, :, :, :] = batch_img
    if (batch_id + 1) % 100 == 0:
        print((batch_id + 1)/100)

torch.save(poisoned_test_samples, 'poisoned_test_samples_WaNet_VGG.pth')
#torch.save(poisoned_test_samples, 'poisoned_test_samples_ISSBA_VGG.pth')
#torch.save(poisoned_test_samples, 'poisoned_test_samples_BadNets_VGG.pth')
#torch.save(poisoned_test_samples, 'poisoned_test_samples_TUAP_VGG.pth')
#torch.save(poisoned_test_samples, 'poisoned_test_samples_PhysicalBA_VGG.pth')
print('finished')
