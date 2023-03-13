"""class containing a model that takes two models andsets them in a sequential order."""
import torch.nn as nn

class Full_Net(nn.Module):
    def __init__(self, model_enc, model_att) -> None:
        super(Full_Net, self).__init__()
        self.alpha = model_enc
        self.beta = model_att

    def forward(self, x):
        x = self.alpha(x)
        x = self.beta(x)
        return x

    def getAlpha(self):
        return self.alpha

    def getBeta(self):
        return self.beta

