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


file_path = 'U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_0/'
# read in the dapi files
all_Z_DAPI = dapi_reader.parallel_dapi_reader(file_path)
# read the detected transcript csv file
detected_transcript = pd.read_csv(
            file_path + 'detected_transcripts.csv')

# # prepare tile
all_z_tiles, fovs = prepare_tile.prepar_tile(detected_transcript, all_Z_DAPI)

# generateing training and validataion set
train_test_set_path = 'U:/Lab/Bereket_public/Merfish_AD_project_data_analysis/cell_pose/cellpose_training_dataset'

CA_DG_fovs = [1637,1668,1669,1670,1671,1672,1688,1689,1690,1691,1692]
CA_DG_fov_idx = [fovs.index(f) for f in CA_DG_fovs]
validation_set = CA_DG_fovs[:4]
start_fov_idx = fovs.index(1668)
end_fov_idx = fovs.index(1695)

with open(train_test_set_path+'/fovs.pickle','wb') as f:
    pickle.dump(fovs,f)

prepare_tile.prepare_train_test_data(all_z_tiles,train_test_set_path,z_idx=0,started_fov_idx=start_fov_idx,end_fov_idx=end_fov_idx)








