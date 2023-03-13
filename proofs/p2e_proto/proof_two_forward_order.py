"""script that contains the proof that the encoder can be run 2 times
until the pass to the decoder happens"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datastruct import ToTensor, WSIDataSet
from model_Enc import model_Enc
from model_Att import model_Att
from model_Full import Full_Net
import copy
from helping_functions import compare_models

import os
#used for reproducibility 
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

#directory where tiles are located
param_tile_dir = os.path.abspath('src/tiles/')

#define the current device of the user
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#constructs the dataset
param_transformed_dataset = WSIDataSet(
                                           root_dir=param_tile_dir,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
#batches the dataset
dataloader = DataLoader(param_transformed_dataset, batch_size=13, shuffle=False)

"""initializes the parameter models"""
param_model_enc = model_Enc()
param_model_att = model_Att()

#initialize the models 
model_partial = Full_Net(param_model_enc, param_model_att)
model_partial_2 = copy.deepcopy(model_partial)

criterion = nn.L1Loss()

optimizer = optim.SGD(model_partial.parameters(), lr= 0.001, nesterov =False)

#set the first fully connected layer depending on the number of parameters 
model_partial.getBeta().set_firstLinearLayer(13312)
model_partial_2.getBeta().set_firstLinearLayer(13312)
model_partial.to('cpu')

#hold one batch of data respectively
param_inputs_first_iter  = None
param_inputs_second_iter = None

#get the first two batches
for i, data in enumerate(dataloader, 0):

    inputs = data['img']
    if i == 0:
        param_inputs_first_iter = inputs
    if  i == 1:
        param_inputs_second_iter = inputs
    if i ==2:
        break

#function that trains a model like in proof_multiple_Encoder_steps.py
def train_model_different_input_order(model, input_1, input_2, optimizer, criterion):
    outputs_old = None
    for i in range(2):
        if i == 0:
            input_1 = input_1.to('cpu')
            outputs_old = model.getAlpha()(input_1)
            del input_1
            torch.cuda.empty_cache()
        if i == 1:
            input_2 = input_2.to('cpu')
            outputs = model.getAlpha()(input_2)
            del input_2
            torch.cuda.empty_cache()
            outputs_old = torch.cat((outputs_old, outputs), 0)

            outputs = model.getBeta()(outputs_old)

            label = torch.ones(1)
            label = label.to('cpu')
            loss = criterion(outputs, label)
            loss.backward()
            loss_model = loss.item()
            optimizer.zero_grad()
            optimizer.step()
            model_out = copy.deepcopy(model).to('cpu').state_dict()
    
    return model_out, loss_model

#contains the model that is trained with the first batch then the second batch 
model_1, loss_1 = train_model_different_input_order(model_partial, param_inputs_first_iter, param_inputs_second_iter, optimizer, criterion)

#used to free some memory
del model_partial
del criterion
del optimizer
torch.cuda.empty_cache()

model_partial_2.to('cpu')
optimizer = optim.SGD(model_partial_2.parameters(), lr = 0.001, nesterov=False)
criterion = nn.L1Loss()

#contains a model that is trained with the second batch first and then the first batch
model_2, loss_2 = train_model_different_input_order(model_partial_2, param_inputs_second_iter, param_inputs_first_iter, optimizer, criterion)

print(loss_1, ' ', loss_2)
"""returns:
example
0.7646514773368835   0.7367379069328308
"""

#compare models 
compare_models(model_1, model_2)
"""returns:
Mismatch found at model 1:  beta.fc1.weight  model 2:  beta.fc1.weight
Mismatch found at model 1:  beta.fc1.bias  model 2:  beta.fc1.bias
"""