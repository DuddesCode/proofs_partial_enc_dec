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

torch.backends.cudnn.deterministic = True
param_tile_dir = os.path.abspath('src/tiles/')
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


param_transformed_dataset = WSIDataSet(
                                           root_dir=param_tile_dir,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
dataloader = DataLoader(param_transformed_dataset, batch_size=13, shuffle=False)

"""initializes the parameter models"""
param_model_enc = model_Enc()
#deepcopy of model_enc
param_model_enc2 = copy.deepcopy(param_model_enc)

param_model_dec = model_Dec()
#deepcopy of model_att
param_model_dec2 = copy.deepcopy(param_model_dec)

 #full model 
model_Full = Full_Net(param_model_enc, param_model_dec).to(global_device)

setup_alpha_pre_train = copy.deepcopy(model_Full).state_dict()
#define criterion, optimizer
criterion = nn.L1Loss()
optimizer = optim.SGD(model_Full.parameters(), lr=0.001, nesterov=False)

running_loss = 0.0
#contains one batch of images
sample = next(iter(dataloader))

subfull_alpha_pre_train = copy.deepcopy(model_Full.getAlpha()).state_dict()
subfull_beta_pre_train = copy.deepcopy(model_Full.getBeta()).state_dict()
#train step for model_Full

#prepare inputs and labels
inputs = sample['img'] 
labels = torch.zeros(1)
labels = labels.type(torch.FloatTensor)
inputs, labels = inputs.to(global_device), labels.to(global_device)

#put them through the full model
outputs = model_Full(inputs)

#compute and backtrack loss of the full model
loss_alpha = criterion(outputs, labels)
loss_alpha.backward()
optimizer.step()
optimizer.zero_grad()
loss_alpha = loss_alpha.item()

#save Full model parts for checks  
setup_alpha_post_train = model_Full.state_dict()
subfull_alpha_post_train = copy.deepcopy(model_Full.getAlpha()).state_dict()
subfull_beta_post_train = copy.deepcopy(model_Full.getBeta()).state_dict()
print('----------------------------------------------------------------------------------')
print('if differences are displayed training has occured in the encoder of the full model')
compare_models(subfull_alpha_post_train, subfull_alpha_pre_train)
print('----------------------------------------------------------------------------------')
print('if differences are displayed training has occured in the decoder of the full model')
compare_models(subfull_beta_post_train, subfull_beta_pre_train)
print('----------------------------------------------------------------------------------')

#used to free some memory
del subfull_alpha_post_train
del subfull_beta_post_train
del subfull_alpha_pre_train
del subfull_beta_pre_train
del inputs
del labels
del model_Full
del optimizer
del outputs
del param_model_dec
del param_model_enc
torch.cuda.empty_cache()

#make the second model ready for training
model_Full_2 = Full_Net(param_model_enc2, param_model_dec2)
model_Full_2.to(global_device)

#saves the untrained model of the partial mode
setup_beta_pre_train = copy.deepcopy(model_Full_2).state_dict()

#define a second optimizer
optimizer_beta = optim.SGD(model_Full_2.parameters(), lr=0.001,nesterov=False)    

#prepare inputs and labels
inputs = sample['img']
labels = torch.zeros(1)
labels = labels.type(torch.FloatTensor)
inputs, labels = inputs.to(global_device), labels.to(global_device)

#save the layers of the second model for checks
subbeta_pre_train = copy.deepcopy(model_Full_2.getBeta()).to('cpu').state_dict()
subalpha_pre_train = copy.deepcopy(model_Full_2.getAlpha()).to('cpu').state_dict()

#pass through Encoder
outputs = model_Full_2.getEncoder()(inputs)

#free memory
del inputs
torch.cuda.empty_cache()

#decoder pass
outputs = model_Full_2.getDecoder()(outputs)
loss_beta = criterion(outputs, labels)

#free memory
del outputs
torch.cuda.empty_cache()

#Full_Net2 loss computation and optimizer
loss_beta.backward()
optimizer_beta.step()
optimizer_beta.zero_grad()
loss_beta = loss_beta.item()

#saves the trained state dict of partial mode
setup_beta_post_train = model_Full_2.state_dict()

subbeta_post_train = model_Full_2.getBeta().to('cpu').state_dict()
subalpha_post_train = model_Full_2.getAlpha().to('cpu').state_dict()


print('Proof that weights and biases are equal if deterministics criterias are satisfied')
#compares model partial mit model Full
compare_models(setup_alpha_post_train, setup_beta_post_train)
print('----------------------------------------------------------------------------------')
print('Proof that the losses are identical')
print(loss_alpha-loss_beta)
print('----------------------------------------------------------------------------------')
print('Proof that the submodels are trained during both loops')
print('if differences are displayed training has occured in the encoder of the partial model')
compare_models(subalpha_post_train, subalpha_pre_train)
print('----------------------------------------------------------------------------------')
print('if differences are displayed training has occured in the decoder of the partial model')
compare_models(subbeta_post_train, subbeta_pre_train)
print('----------------------------------------------------------------------------------')