"""class that contains an experimental WSI Dataset and Dataloader"""
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

import torchvision.transforms as transforms

class WSIDataSet(Dataset):
    """Dataset to be used in all proofs

    Parameters
    ----------
    Dataset : Dataset
        base class to be used
    
    Attributes
    ----------
    root_dir : string
        contains the root directory
    transform : function
        used to transform the data of the dataset
    """    
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        """returns the length of the dataset

        _extended_summary_

        Returns
        -------
        int
            contains the length of the dataset
        """        
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        """returns the item at the requested position idx
        Parameters
        ----------
        idx : int
            contains the requested position
        
        Returns
        ---------
        sample : dict
            contains the tile as a transformed image
        """        
        sample = {'tile':Image.open(os.path.join(self.root_dir, os.listdir(self.root_dir)[idx]))}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """transforms an object to a Tensor

    Parameters
    ----------
    object : dict of form {'tile: img'}
        contains an image that has to be transformed to a Tensor
    
    Functions
    ---------
    __call__(sample)
        performs the tensor transformation to the object
    """    
    def __call__(self, sample):
        """transforms the sample to a Tensor

        Parameters
        ----------
        sample : dict {'tile': Image}
            contains a dict that contains an image

        Returns
        -------
        dict
            contains the image transformed to a tensor
        """                
        image = sample['tile']
        image = np.array(image)
        image = image.astype(np.float)
        image = image.transpose((2,0,1))

        return {'img': (torch.from_numpy(image)).type(torch.float32)}

