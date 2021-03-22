#!/usr/bin/env python
# coding: utf-8

# # Import the modules

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm_notebook as tqdm

torch.manual_seed(0)
np.random.seed(0)

from utilities.dataset import *
from utilities.train_and_fit_3class import *
from utilities.model_maxout_3class import *


# # Set various settings

# In[ ]:


dataset_path = r'E:\audio_library'
load_previous_dataset = False
target_column = 'id-type'
target_value_name = 'thrushes'
labelmapping = {'RE-F': 1, 'B_-F': 2, 'ST-F':3}
model_arch = 'maxout'
batch_size = 32
epochs = 50
lr = 0.0001
weight_decay = 0.2


# # Compile the dataset based on the audio clips
# 
# Create a dictionary containing the settings and to hold the results of each epoch
# 
# Create the training and validation subsets and create batch loaders

# In[ ]:


#where to save the model?
model_name = model_arch + '_b' + str(batch_size) + '_e' + str(epochs) + '_lr' + str(lr) + '_wd' + str(weight_decay)
savedir = model_name
if not os.path.isdir(savedir):
    os.mkdir(savedir)

#make the checkpoint dictionary with default values    
outpath = os.path.join(savedir, 'checkpoint.tar')
checkpoint_dict = {'filename': outpath,
                   'sr': 22050,
                   'n_classes': 3,
                   'epoch': 0,
                   'loss': np.inf,
                   'training_loss': [],
                   'validation_loss': [],
                   'model_state_dict': None,
                   'optimizer_state_dict': None}

# use GPU if available (unless use_cuda is False)
device = get_and_print_device(use_cuda=True)

# create an empty dataset
dataset = Dataset()

# either compile a dataset or load a previously saved one
if not load_previous_dataset:
    assert dataset_path is not None, 'dataset_path must be provided'
    # compile the dataset by extracting audio clips
    dataset.compile(path=dataset_path)
    # map 'id-type' to target
    dataset.map_target(column=target_column, mapping=labelmapping)
    dataset.tabulate_targets()
else:
    # load a previously saved dataset
    dataset.load()
    dataset.tabulate_targets()

# update checkpoint_dict with sampling rate and number of classes
checkpoint_dict['sr'] = dataset.sr
checkpoint_dict['n_classes'] = dataset.num_classes
checkpoint_dict['mapping'] = dataset._mapping

# split dataset into training (80%) and validation (20%) subsets
training_set, validation_set = dataset.random_split_stratified_by_target(class_fraction=0.2)

# create batching data loaders for training and validation data
training_generator = get_training_set_generator(dataset, training_set, batch_size, stratified=True)
validation_generator = get_validation_set_generator(validation_set, 2*batch_size)


# # Run the model
# 
# Fit the defined model and calculate basic precision-recall stats
# 
# Display some examples of the fitted functions
# 
# Save the model

# In[ ]:


#call the model definition
#use fmin to set minimum frequency for the Melspectrogram
model = Conv1D_Maxout(sr=checkpoint_dict['sr'], fmin = 3000).to(device)

# set up an optimizer
params = list(model.parameters())
optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

# main training loop
fit(epochs, model, optimizer, training_generator, validation_generator, device, checkpoint_dict)
plot_losses(checkpoint_dict['training_loss'], checkpoint_dict['validation_loss'])
csvname = os.path.join(savedir, 'losses.csv')
save_losses(checkpoint_dict['training_loss'], checkpoint_dict['validation_loss'], csvname)

# load best model according to validation data
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
checkpoint = torch.load(checkpoint_dict['filename'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print('best model recorded at epoch {} with loss={}'.format(checkpoint['epoch'], checkpoint['loss']))

# plot precision and recall for the validation clips
model.eval()
#plot_precision_recall(model, validation_generator, device)
plot_precision_recall(model, validation_generator, device, channel = 0)
plot_precision_recall(model, validation_generator, device, channel = 1)
plot_precision_recall(model, validation_generator, device, channel = 2)

plot_top_losses(model, validation_set, device, checkpoint_dict)

# save the trained model in a format that can be loaded in C++ for inference
model = model.cpu().eval()
torchscript_module = torch.jit.script(model)
modname = 'model_' + target_column + '=' + target_value_name + '.pt'
outpath = os.path.join(savedir, modname)
torchscript_module.save(outpath)
print('model saved to {}'.format(outpath))


# In[ ]:




