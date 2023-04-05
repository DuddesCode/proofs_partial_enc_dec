"""class containing a model that takes two models andsets them in a sequential order."""
import torch.nn as nn

class Full_Net(nn.Module):
    """defines the complete Network

    Parameters
    ----------
    nn : nn.Module
        base class from which the functions are derived
    
    Functions
    ----------
    forward(x)
        defines how the data is passed trough the model
    getEncoder()
        returns the Encoder
    getDecoder()
        returns the Decoder
    """    
    def __init__(self, model_enc, model_att) -> None:
        super(Full_Net, self).__init__()
        self.encoder = model_enc
        self.decoder = model_att

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
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x

    def getEncoder(self):
        """returns the Encoder

        Returns
        -------
        nn.Module
            Encoder model
        """        
        return self.encoder

    def getDecoder(self):
        """returns the Decoder

        Returns
        -------
        nn.Module
            Decoder model
        """        
        return self.decoder

