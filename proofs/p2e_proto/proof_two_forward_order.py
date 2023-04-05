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
param_model_dec = model_Dec()

#initialize the models 
model_partial = Full_Net(param_model_enc, param_model_dec)
model_partial_2 = copy.deepcopy(model_partial)

criterion = nn.L1Loss()
#define optimizer
optimizer = optim.SGD(model_partial.parameters(), lr= 0.001, nesterov =False)

#set the first fully connected layer depending on the number of parameters 
model_partial.getDecoder().set_firstLinearLayer(13312)
model_partial_2.getDecoder().set_firstLinearLayer(13312)
model_partial.to('cpu')

#hold one batch of data respectively
param_inputs_first_iter  = None
param_inputs_second_iter = None

#get the first two batches preventing nondeterminism
for i, data in enumerate(dataloader, 0):

    inputs = data['img']
    if i == 0:
        param_inputs_first_iter = inputs
    if  i == 1:
        param_inputs_second_iter = inputs
    if i ==2:
        break

def train_model_different_input_order(model, input_1, input_2, optimizer, criterion):
    """trains a model with different inputs

    used to proof that multiple encoder steps are possible before the decoder step

    Parameters
    ----------
    model : Full_Net
        the model that is trained
    input_1 : batch of Dataloader
        contains the first batch to be trained
    input_2 : batch of the Dataloader
        contains the second batch to be trained
    optimizer : torch.optimizer
        an optimizer used to train the model
    criterion : loss_fn
        used to compute the loss of the model

    Returns
    -------
    Full_Net
        contains the trained model
    float
        contains the loss of the model
    """    
    outputs_old = None
    #run for two iterations
    for i in range(2):
        #get outputs_old in the first iteration
        if i == 0:
            input_1 = input_1.to('cpu')
            outputs_old = model.getEncoder()(input_1)
            del input_1
            torch.cuda.empty_cache()
        #concatenate the first two inputs
        if i == 1:
            input_2 = input_2.to('cpu')
            outputs = model.getEncoder()(input_2)
            del input_2
            torch.cuda.empty_cache()
            outputs_old = torch.cat((outputs_old, outputs), 0)
            #decoder pass
            outputs = model.getDecoder()(outputs_old)
            #compute loss and backpropagate through the entire model
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
print("losses of the models")
print(loss_1, ' ', loss_2)

#compare models 
compare_models(model_1, model_2)