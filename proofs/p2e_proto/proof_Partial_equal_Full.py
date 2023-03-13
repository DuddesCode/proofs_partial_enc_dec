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

param_model_att = model_Att()
#deepcopy of model_att
param_model_att2 = copy.deepcopy(param_model_att)

#pAlpha
#alpha
#mAlpha

model_Full = Full_Net(param_model_enc, param_model_att).to(global_device)

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

inputs = sample['img'] 
labels = torch.zeros(1)
labels = labels.type(torch.FloatTensor)
inputs, labels = inputs.to(global_device), labels.to(global_device)

outputs = model_Full(inputs)

loss_alpha = criterion(outputs, labels)
loss_alpha.backward()
optimizer.step()
optimizer.zero_grad()
loss_alpha = loss_alpha.item()


setup_alpha_post_train = model_Full.state_dict()
subfull_alpha_post_train = copy.deepcopy(model_Full.getAlpha()).state_dict()
subfull_beta_post_train = copy.deepcopy(model_Full.getBeta()).state_dict()
#used to free some gpu space
print('----------------------------------------------------------------------------------')
print('model Full Encoder')
compare_models(subfull_alpha_post_train, subfull_alpha_pre_train)
print('----------------------------------------------------------------------------------')
print('model Full Decoder')
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
del param_model_att
del param_model_enc
torch.cuda.empty_cache()


model_Full_2 = Full_Net(param_model_enc2, param_model_att2)
model_Full_2.to(global_device)

#saves the untrained model of the partial mode
setup_beta_pre_train = copy.deepcopy(model_Full_2).state_dict()

optimizer_beta = optim.SGD(model_Full_2.parameters(), lr=0.001,nesterov=False)    

inputs = sample['img']

labels = torch.zeros(1)
labels = labels.type(torch.FloatTensor)
inputs, labels = inputs.to(global_device), labels.to(global_device)

subbeta_pre_train = copy.deepcopy(model_Full_2.getBeta()).to('cpu').state_dict()
subalpha_pre_train = copy.deepcopy(model_Full_2.getAlpha()).to('cpu').state_dict()

outputs = model_Full_2.getAlpha()(inputs)

del inputs
torch.cuda.empty_cache()

outputs = model_Full_2.getBeta()(outputs)
loss_beta = criterion(outputs, labels)

del outputs
torch.cuda.empty_cache()

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
print('model partial Encoder')
compare_models(subalpha_post_train, subalpha_pre_train)
print('----------------------------------------------------------------------------------')
print('model partial Decoder')
compare_models(subbeta_post_train, subbeta_pre_train)
print('----------------------------------------------------------------------------------')
'''l1 = criterion(model_Full(inputs),labels)
l2 = criterion(model_Full_2(inputs),labels)
print(l1,l2)'''