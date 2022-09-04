
import os, shutil
import numpy as np
import glob
import matplotlib.pyplot as plt
from  cellpose import core,utils,io,models, metrics
from os import listdir
from natsort import natsorted
import dapi_reader
import prepare_tile
import pandas as pd
import pickle

use_GPU = core.use_gpu()
yn = ['NO','Yes']
print(f'>> GPU activated? {yn[use_GPU]}')
train_test_set_path = 'U:/Lab/Bereket_public/Merfish_AD_project_data_analysis/cell_pose/cellpose_training_dataset'

with open(train_test_set_path+'/fovs.pickle','rb') as f:
    fovs = pickle.load(f)

import pdb
pdb.set_trace()


CA_DG_fovs = [1637,1668,1669,1670,1671,1672,1688,1689,1690,1691,1692]
CA_DG_fov_idx = [fovs.index(f) for f in CA_DG_fovs]
validation_set = CA_DG_fovs[:4]
start_fov_idx = fovs.index(1668)
end_fov_idx = fovs.index(1695)

# split the training and validation set
prepare_tile.split_train_test(train_test_set_path,validation_fovs = validation_set)

all_train_files = listdir (train_test_set_path + '/training_set')
all_test_files = listdir(train_test_set_path + "/test_set")


train_seg = []
train_files = []

for t_file in all_train_files:
    if t_file.split(',')[-1] == 'npy':
        train_seg.append(train_test_set_path + '/training_set/' + t_file)
    else:
        train_files.append(train_test_set_path + '/training_set/' + t_file)


# do the same for the testing files

test_seg = []
test_files = []

for t_file in all_test_files:
    if t_file.split(',')[-1] == 'npy':
        train_seg.append(train_test_set_path + '/training_set/' + t_file)
    else:
        train_files.append(train_test_set_path + '/training_set/' + t_file)


# specify training parameters

train_dir = train_test_set_path + '/training_set/'
test_dir = train_test_set_path + "/test_set/"
initial_model = 'U:/Lab/Bereket_public/Merfish_AD_project_data_analysis/cell_pose/model_AD_net/cellpose_AD_net_71122'
# path to initial model

model_name = 'cellpose_AD_net_9222'
n_epochs = 250
chan = 0
chan2 = 0
learning_rate = 0.1
weight_decay = 0.0001



# Train the new model

# start logger
logger = io.logger_setup()

# define cellpose model (without size model)
model = models.CellposeModel(gpu = use_GPU, model_type = initial_model)

# set channels
channels = [chan,chan2]

# get the files
output = io.load_train_test_data(train_dir,test_dir,mask_filter = "_seg_.npy")
train_data, train_labels,_,test_data,test_labels,_ = output

new_model_path = model.train(train_data,train_labels,
                             test_data = test_data,
                             test_labels = test_labels,
                             channels = channels,
                             save_path = train_dir,
                             n_epochs = n_epochs,
                             learning_rate=learning_rate,
                             weight_decay=weight_decay,
                             nimag_per_epoch = 8, # this is basically the batch size
                             model_name= model_name
                             )
# diameter of lavels in trainng images
diam_labels = model.dim_labels.copy()

