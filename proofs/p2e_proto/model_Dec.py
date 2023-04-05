"""classs that contains the test Encoder Model"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class model_Dec(nn.Module):
    """defines the Decoder Model used in the proofs

    Parameters
    ----------
    nn : nn.Module
        base class from which functions are derived
    
    Attributes
    ----------
    fc1 : nn.Linear
    fc2 : nn.Linear
    fc3 : nn.Linear
    fc4 : nn.Linear
    fc5 : nn.Linear
    
    Functions
    ----------
    forward(x)
        defines how data is passed through the model
    setFirstLinearLayer(layer)
        sets the first fully connected layer to the desired input
    """    
    def __init__(self):
        super(model_Dec, self).__init__()
        self.fc1 = nn.Linear(6656, 6144)
        self.fc2 = nn.Linear(6144, 384)
        self.fc3 = nn.Linear(384, 96)
        self.fc4 = nn.Linear(96, 48)
        self.fc5 = nn.Linear(48,1)

    def forward(self, x):
        """returns the data after it has been passed through

        Parameters
        ----------
        x : data
            contains a batch of images of tiles transformed to Tensors

        Returns
        -------
        x
            data after it has been passed through the model
        """        
        temp  = x
        temp = temp.view(-1,1)  
        temp = torch.flatten(temp)
        x = F.relu(self.fc1(temp))
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x

    def set_firstLinearLayer(self, layer):
        """sets the first fully connected layer of the model to the desired layer

        Parameters
        ----------
        layer : int
            contains the number of parameters that have to be set
        """        
        self.fc1 = nn.Linear(layer, 6144)