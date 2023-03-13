"""class that contains an experimental WSI Dataset and Dataloader"""
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

import torchvision.transforms as transforms

class WSIDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        
        sample = {'tile':Image.open(os.path.join(self.root_dir, os.listdir(self.root_dir)[idx]))}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image = sample['tile']
        image = np.array(image)
        image = image.astype(np.float)
        #print(image.shape)
        image = image.transpose((2,0,1))
        return {'img': (torch.from_numpy(image)).type(torch.float32)}

