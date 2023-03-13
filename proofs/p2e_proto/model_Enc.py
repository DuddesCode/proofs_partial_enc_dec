"""classs that contains the test Encoder Model"""
import torch.nn as nn
import torch.nn.functional as F



class model_Enc(nn.Module):
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
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
    
    def forward(self, x):
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