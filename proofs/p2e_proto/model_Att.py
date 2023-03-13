"""classs that contains the test Encoder Model"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class model_Att(nn.Module):
    def __init__(self):
        super(model_Att, self).__init__()
        self.fc1 = nn.Linear(6656, 6144)
        self.fc2 = nn.Linear(6144, 384)
        self.fc3 = nn.Linear(384, 96)
        self.fc4 = nn.Linear(96, 48)
        self.fc5 = nn.Linear(48,1)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        temp  = x
        temp = temp.view(-1,1)  
        temp = torch.flatten(temp)
        ''' x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))'''
        #x = torch.flatten(x,1)
        x = F.relu(self.fc1(temp))
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def set_firstLinearLayer(self, layer):
        self.fc1 = nn.Linear(layer, 6144)