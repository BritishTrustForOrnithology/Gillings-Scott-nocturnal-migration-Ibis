#!/usr/bin/env python
# coding: utf-8

# # Import the modules

# In[ ]:


import librosa
import numpy as np
import torch
from scipy.signal import find_peaks
import os
import pandas as pd
from tqdm import tqdm_notebook as tqdm

# import the model definition
from utilities.model_maxout_3class import *


# # Define a dataset for batching the data to be processed

# In[ ]:


class InferenceDataset(torch.utils.data.Dataset):
    
    # frame_length = sr * max_duration
    def __init__(self, x, frame_length):
        frame_length = min(x.size, frame_length)
        rem = x.size % frame_length
        if (rem != 0):
            x = np.pad(x, (0,rem), mode='constant')
        self.frames = librosa.util.frame(np.asfortranarray(x), 
                                         frame_length=frame_length, 
                                         hop_length=frame_length, 
                                         axis=-1).T

    def __len__(self):
        return self.frames.shape[0]
    
    def __getitem__(self, idx):
        return self.frames[idx]


# # Method to apply the model to the dataset

# In[ ]:


def apply_model(wdir, use_cuda=True, mapping=None, thresholds=None, gapseconds=None, export_allscores=False):

    #where to run the model
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #device = torch.device("cpu")

    #how many labels (aka species)
    num_channels = len(set(mapping.values()))
    
    #how many classes (i.e. species + 1 for nil)
    num_classes = num_channels + 1
    
    #get the names of classes for labels
    ids = []
    for channel in range(num_channels):
        id = [k for k,v in mapping.items() if v == (channel+1)]
        id = '|'.join(id)
        ids.append(id)
        
    # create a model object - same settings as used in training!
    model = Conv1D_Maxout(sr = 22050, fmin = 3000)

    # loading model weights from tar file
    savedir = r'E:\python\savedmodels\'
    name = 'checkpoint'
    checkpoint_file = os.path.join(savedir, name + '.tar')
    checkpoint_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    model = model.to(device).eval()

    # effective hop is spectrogram hop_length * model downsampling factor
    # model downsampling factor should be 1 for a dilated model with no pooling layers
    effective_hop = 32 * 1
    
    # duration of analysis clips in seconds
    max_duration = 60
    
    #using the distances (in time) between valid peaks for each species, determine what distances
    #will be in samples
    distances = librosa.time_to_frames(np.array(gapseconds), sr=22050, hop_length=effective_hop)
    
    # find wav and flac files in wdir
    files = librosa.util.find_files(wdir, ext=['wav', 'flac'], recurse=True)
    for file in tqdm(files):
        # load the whole file in one go (mono for simplicity)
        x, sr = librosa.load(file, sr=22050, res_type='kaiser_fast')
        
        # InferenceDataset splits the file into chunks
        dataset = InferenceDataset(x, sr*max_duration)
        
        # create a generator for iterating over the split data
        generator = torch.utils.data.DataLoader(dataset)
        
        allscores = np.array([]).reshape(0,3)
        offsets = []
        labels = []
        clip_start = 0.0
        
        # process file in chunks fetched by the generator
        for local_batch in generator:
            logits = model(local_batch.to(device))
            
            batch_scores = logits.sigmoid().cpu().detach().numpy()
            
            for batch in range(batch_scores.shape[0]):
                for channel in range(num_channels):
                    batch_peaks, _ = find_peaks(batch_scores[batch,channel], height=thresholds[channel], distance=distances[channel])
                    peak_times = librosa.frames_to_time(batch_peaks, sr=sr, hop_length=effective_hop) + clip_start
                    n_peaks = len(peak_times)
                    
                    #print(type(peak_times))
                    label = np.full(shape = n_peaks, fill_value=ids[channel])
                    offsets.append(peak_times)
                    labels.append(label)
                clip_start += max_duration

            # stack all scores for this file 
            #squeeze off single dimension
            batch_scores = np.squeeze(batch_scores)
            #transpose from wide to long
            batch_scores = np.transpose(batch_scores)
            #vertical stack/append
            allscores = np.vstack([allscores, batch_scores])
            
        #round
        allscores = np.around(allscores,3)

        if export_allscores:
            outfile = os.path.splitext(file)[0] + '.scores'
            pd.DataFrame(allscores).to_csv(outfile, sep=',', header=None, index=None)
        
        # stack offsets of all detected calls for this file
        offsets = np.hstack(offsets)
        labels = np.hstack(labels)
        #round offsets
        offsets = np.around(offsets,3)
        if offsets.size:
            # if calls detected in file print file and call count
            print('{}: {}'.format(os.path.basename(file), offsets.size))
            
            # write detections to an Audacity style labels file
            outfile = os.path.splitext(file)[0] + '_predsREB_ST.txt'
            pd.DataFrame({'start': offsets, 'end': offsets, 'labels': labels}).to_csv(outfile, sep='\t', header=None, index=None)


# # Apply the prediction method to a folder of audio files

# In[ ]:


# directory of audio files to be processed
wdir = r'E:\new_recordings'

#the mapping as used in training
mapping = {'RE-F': 1, 'B_-F':2, 'ST-F':3}

#score thresholds to be used for creating detections
thresholds = [0.57, 0.75, 0.95] #these based on f1 max threhsolds

#gap in seconds between valid detections
gapseconds = [0.1, 0.1, 0.1]

#get the predictions
apply_model(wdir = wdir, mapping = mapping, thresholds = thresholds, gapseconds = gapseconds, export_allscores = True)


# In[ ]:




