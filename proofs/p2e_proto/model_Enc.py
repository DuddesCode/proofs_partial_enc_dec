"""classs that contains the test Encoder Model"""
import torch.nn as nn
import torch.nn.functional as F



class model_Enc(nn.Module):
    """Encoder Model used in the proofs

    Parameters
    ----------
    nn : nn.Module
        Module from which the model is derived
    
    Attributes
    ----------
    conv1 : nn.Conv2d
    pool1 : nn.MaxPool2d
    conv2 : nn.Conv2d
    conv3 : nn.Conv2d
    pool2 : nn.MaxPool2d
    conv4 : nn.Conv2d
    conv5 : nn.Conv2d
    pool3 : nn.MaxPool2d
    pool4 : nn.MaxPool2d

    Functions
    ----------
    forward(x)
        defines how the data is passed through the model
    """    
    def __init__(self):
        super(model_Enc, self).__init__()
        self.conv1 = nn.Conv2d(4, 40, 1)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(40, 56, 1)
        self.conv3 = nn.Conv2d(56, 32, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32,16, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16,8,1)
        self.pool4 = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        """defines how the data is passed throught the model

        Parameters
        ----------
        x : batch of tensors
            contains a batch of tensors 

        Returns
        -------
        batch of tensors
            contains the transformed data
        """        
        x = F.relu(self.conv1(x))

        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)

        return x