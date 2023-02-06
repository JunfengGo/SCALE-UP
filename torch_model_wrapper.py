'''
This is the code of obtaining samples from a given dataloader and save them as a tensor.
'''

import os
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
import torchvision.models as models
import time
import argparse
from resnet import *
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--gpu-id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu_id
#dataloader_root_dir = 'WaNet/poisoned_test_samples_WaNet.pth'
dataloader_root_dir = 'benign_test_samples.pth'


device = torch.device('cuda:0')
resnet18 = models.resnet18(pretrained=False)
resnet18.fc = torch.nn.Linear(512, 200)
model = resnet18
model.load_state_dict(torch.load("BadNets/ckpt_epoch_200.pth",map_location=device))
model.to(device)
model.eval()

poisoned_test_samples = torch.load(dataloader_root_dir,map_location=device)

#adding random noise

poisoned_test_samples = poisoned_test_samples+0.02*torch.rand(size=poisoned_test_samples.shape,device=device)

labels = torch.load("benign_labels.pth")



decisions = np.empty((10000,11))

for i in range(100):

    img_batch = poisoned_test_samples[i*100:(i+1)*100]
    img_batch.to(device)
    #evals = 0.1*torch.randn(100, 3, 32, 32,device=device)

    for h in range(1,12):

        img_batch_re = torch.clamp(h*img_batch,0,1)
        decisions[i*100:(i+1)*100,(h-1)] = torch.max(model(img_batch_re),1)[1].detach().cpu().numpy()
print(decisions)
print(np.mean(decisions[:,0]==np.reshape(labels.numpy(),10000)))
a = decisions[decisions[:,0]==np.reshape(labels.numpy(),10000)]
print(a.shape)
np.save("saved_np/BadNets/tiny_benign.npy",decisions)



