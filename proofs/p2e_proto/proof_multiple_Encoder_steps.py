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
from model_Dec import model_Dec
from model_Full import Full_Net
import copy
from helping_functions import compare_models

import os
#used for reproducability of the training
torch.backends.cudnn.deterministic = True


param_tile_dir = os.path.abspath('src/tiles/')
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


param_transformed_dataset = WSIDataSet(
                                           root_dir=param_tile_dir,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
dataloader = DataLoader(param_transformed_dataset, batch_size=13, shuffle=False)

#initializes the parameter models
param_model_enc = model_Enc()
param_model_dec = model_Dec()

model_partial = Full_Net(param_model_enc, param_model_dec)

criterion = nn.L1Loss()

optimizer = optim.SGD(model_partial.parameters(), lr= 0.001, nesterov =False)

#used to fit the first linear layer to the concatenated input
model_partial.getDecoder().set_firstLinearLayer(13312)
model_partial.to(global_device)

model_partial_pre_run = copy.deepcopy(model_partial).to('cpu').state_dict()
"""used for the training loop"""
for i, data in enumerate(dataloader, 0):

    inputs = data['img']
    labels = torch.ones(1)

    inputs, labels = inputs.to(global_device), labels.to(global_device)

    if i == 0:
        inputs_first_enc = inputs.to('cpu')
        print('first pass through encoder')
        outputs = model_partial.getEncoder()(inputs)
        outputs_old = outputs
    
    if i == 1:
        inputs_second_run = inputs.to('cpu')
        outputs = model_partial.getEncoder()(inputs)
        print('second pass through encoder')
        #need to use not due to the nature of torch.equal
        print("Are the inputs of first encoder run different than the second: ", not torch.equal(inputs_first_enc, inputs_second_run))
        print('----------------------------------------------------------------------------------')
        print('Are the outputs of each iteration different: ', not torch.equal(outputs_old, outputs))
        print('----------------------------------------------------------------------------------')
        outputs_old = torch.cat((outputs_old, outputs),0)
        outputs_old.retain_grad()
        print('pass through decoder')
        
        outputs = model_partial.getDecoder()(outputs_old)
        outputs.retain_grad()
        loss= criterion(outputs, labels)
        print('loss is being computed')
        loss.backward(retain_graph=True)
        print('----------------------------------------------------------------------------------')
        print('proof that gradients of the concatenated batches are computed:')
        print(outputs_old.grad)
        print('----------------------------------------------------------------------------------')
        print('proof that the overall gradient was computed: shows the gradients of the final output')
        print(outputs.grad)
    
        optimizer.step()
        optimizer.zero_grad()
        model_partial_post_run = copy.deepcopy(model_partial).to('cpu').state_dict()
        loss_x2 = loss.item()
        print('----------------------------------------------------------------------------------')
        print('show the difference in the model pre and post run')
        compare_models(model_partial_post_run, model_partial_pre_run)
        print('----------------------------------------------------------------------------------')
        print('current loss')
        print(loss_x2)
    


    if i == 2:
        sys.exit()

    