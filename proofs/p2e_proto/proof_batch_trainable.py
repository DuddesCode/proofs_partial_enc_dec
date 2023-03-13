"""scrit that contains the proof that the model can be run in a partial full trainable way.
it is decided by the batch whether it is is fully end to end trained or not.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datastruct import ToTensor, WSIDataSet
from model_fitted_encoder import model_fitted_Enc
from model_fitted_decoder import model_fitted_Decoder
from model_Full import Full_Net
import copy
import os
from progress.bar import IncrementalBar
from helping_functions import compare_models

#used for reproducibility 
#torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)

#directory where tiles are located
param_tile_dir = 'src/tiles/'

#define the current device of the user
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#constructs the dataset
param_transformed_dataset = WSIDataSet(
                                           root_dir=os.path.abspath(param_tile_dir),
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
#batches the dataset
dataloader = DataLoader(param_transformed_dataset, batch_size=13, shuffle=False)

"""initializes the parameter models"""
param_model_enc = model_fitted_Enc()
param_model_att = model_fitted_Decoder()

#initialize the models 
model_partial = Full_Net(param_model_enc, param_model_att)
#model_partial_2 = copy.deepcopy(model_partial)

criterion = nn.L1Loss()

optimizer = optim.SGD(model_partial.parameters(), lr= 0.001, nesterov =False)

model_partial.to(global_device)

#definition of train loop

#variable to hold the overall encoded image
outputs_old = None
marked_output = None
unmarked_output = None

bar = IncrementalBar('encoding image ', max=len(os.listdir(os.path.abspath(param_tile_dir))) / 13)

for i, data in enumerate(dataloader, 0):

    inputs = data['img']
    inputs = inputs.to(global_device)

    #the first three batches are sent with gradient
    if i <= 3:
        mark = True
    else:
        mark = False
    
    #if a batch is marked. it is pushed through the encoder with gradient
    if mark:
        outputs = model_partial.getAlpha()(inputs)
        marked_output = outputs
        marked_output.retain_grad()
    #if it is not it is sent with torch.no_grad
    else:
        with torch.no_grad():
            outputs = model_partial.getAlpha()(inputs)
            print(outputs.grad_fn)
    outputs.to(global_device)

    #saves an unmarked outputs to check if it has a gradient
    if i == 4:
        unmarked_output = outputs

    #used to concatenate the outputs of the encoder    
    if i == 0:
        outputs_old = outputs
        outputs_old.to(global_device)
        
    else:
        outputs_old = torch.cat((outputs_old, outputs), 0)
        outputs_old.to(global_device)

    #stops after 6 Encoder steps
    if i == 6:
        break
    bar.next()
bar.finish()
print(outputs_old)
#pushes the concatenated outputs through the Decoder
outputs = model_partial.getBeta()(outputs_old)
#used so that the gradient is not deleted from the output
outputs.retain_grad()

label = torch.ones(1)
label = label.to(global_device)

loss = criterion(outputs, label)
#loss computation for the whole model
loss.backward()

print('----------------------------------------------------------------------------------')
print('Gradient of the final output of the Decoder')
print(outputs.grad)
print('----------------------------------------------------------------------------------')
print('gradient of a sample marked batch')
print(marked_output.grad)
print('----------------------------------------------------------------------------------')
print('gradient of an unmarked batch')
print(unmarked_output.grad)

loss_model = loss.item()

#optimizer steps
optimizer.zero_grad()
optimizer.step()
print('----------------------------------------------------------------------------------')
print('loss of the model')
print(loss_model)
