"""fitted Decoder Model to use in proof_batch_trainable as it needs a smaller dimension"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class model_fitted_Decoder(nn.Module):
    def __init__(self):
        super(model_fitted_Decoder, self).__init__()
        self.fc1 = nn.Linear(2912, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8,1)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
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
        self.fc1 = nn.Linear(layer, 6144)