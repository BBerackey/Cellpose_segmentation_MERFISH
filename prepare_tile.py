import pdb

import pandas as pd
import numpy as np
from skimage import exposure
from PIL import Image
from skimage.io import imsave, imread
import os
from cellpose import io # cellpose input output module

def prepar_tile(detected_transcript,all_Z_DAPI):
    # tile the image based on the x,y max and min for each fov
    detected_transcript['fov'] = detected_transcript.fov.astype('category')
    fovs = list(detected_transcript.fov.cat.categories)
    x_bounds = np.empty([len(fovs), 2])  # [[x_min,x_max]]
    y_bounds = np.empty([len(fovs), 2])  # [[y_min,y_max]]
    for i, f in enumerate(fovs):
        temp_df = detected_transcript[detected_transcript['fov'] == f]
        x_bounds[i, 0], x_bounds[i, 1] = temp_df.global_x.min(), temp_df.global_x.max()
        y_bounds[i, 0], y_bounds[i, 1] = temp_df.global_y.min(), temp_df.global_y.max()

    # convert the xy max min measures into pixel
    min_x = detected_transcript.global_x.min()
    min_y = detected_transcript.global_y.min()
    x_bounds_pxl = ((x_bounds - min_x) * (1 / 0.108)).astype('int')
    y_bounds_pxl = ((y_bounds - min_y) * (1 / 0.108)).astype('int')

    all_z_tiles = []

    for z_idx in range(7):
        dapi_tile_list = []
        for idx in range(x_bounds_pxl.shape[0]):
            xpxl_min, xpxl_max, ypxl_min, ypxl_max = x_bounds_pxl[idx, 0], x_bounds_pxl[idx, 1], y_bounds_pxl[idx, 0], \
                                                     y_bounds_pxl[idx, 1]
            dapi_tile_list.append(
                all_Z_DAPI[z_idx][ypxl_min:ypxl_max, xpxl_min:xpxl_max])  # x for detected transcript is y in the pxl
            # * CHECK the order of x and y
        all_z_tiles.append(dapi_tile_list)

    return all_z_tiles, fovs

def prepare_train_test_data(all_z_tiles,save_path, **kwargs):
    """
    This function will create train and validation sets based on optionally specific fov start index

    :param all_z_tiles: list of dapi tiles fo reach z layer
    :param save_path: path to save the files
    :param kwargs:  optional argument specifing start fov idx. kwargs is a dictionary

    :return: None
    """

    if len(kwargs) == 0:
        # if no specification about the fov is given pick random tiles
        num = np.random.randint(100, 200, size=[1, 100])
        dapi_tiles = all_z_tiles[0]
        z_idx = 0
    elif ('z_idx' in kwargs) & ('started_fov_idx' in kwargs ):
        # start from the specific fov_idx and pick the next 100 frames for training
        for kw in kwargs.keys():
            if kw == 'started_fov_idx':
               started_fov_idx  = kwargs[kw]

            elif kw == 'end_fov_idx':
                end_fov_idx = kwargs[kw] -  started_fov_idx
            else:
                z_idx = kwargs[kw]
                end_fov_idx = 100

        num = np.random.randint(started_fov_idx,started_fov_idx + end_fov_idx, size=[1, end_fov_idx])
        dapi_tiles = all_z_tiles[z_idx]

    selected_dapi_tile = [dapi_tiles[i] for i in num[0]]

    # save the files
    if not os.path.isdir(save_path + 'all_selected_tiles/'):
        os.mkdir(save_path + 'all_selected_tiles/')

    for i,dapi_imag in enumerate(selected_dapi_tile):
        imsave(save_path + 'all_selected_tiles/tile_' + str(z_idx) + '_' + str(i) + '.tif', dapi_imag)

    return None


def split_train_test(file_path,split_percent = 0.25,start_fov_idx=0, validation_fovs = None):
        """

        :param file_path: file path where the data is saved
        :param split_percent: percentage specifying how much to use for validation
        :param validation_fovs: this specifies the list of fovs that we want to used for validation
        :return:

        """

        # read in the .npy and .tif files
        all_tif_files = []
        all_npy_files = []

        for files in os.listdir(file_path):

             if files.endswith('.npy'):
                 # this npy file is a dictionary with the mask and corresponding tif file
                 labeled_image = np.load(file_path + '/' + files, allow_pickle=True).item()
                 all_tif_files.append(labeled_image['img'])
                 all_npy_files.append(labeled_image)

        # just to check if the length of the two list is the same
        if len(all_tif_files) != len(all_npy_files):
            index_limit = np.minimum(len(all_tif_files),len(all_npy_files))
        else:
            index_limit = len(all_tif_files)

        # randomly split the selected_dapi_tile into train set and 25% into validation set
        rand_idx = np.random.randint(start_fov_idx,start_fov_idx+index_limit,size = [1,int(0.25*index_limit)])
        rand_idx_list = list(set(rand_idx.tolist()[0])) # converting the np array to list and making sure the list is unqie

        if validation_fovs != None:
            for val in validation_fovs:
                if val not in rand_idx_list:
                    rand_idx_list.append(val)  # making sure the validation fov will be part of the randomly selected validation set

        valid_count = 0
        train_count = 0
        if not os.path.isdir(file_path + '/validation_set/'):
            os.mkdir(file_path + '/validation_set/')
        if not os.path.isdir(file_path + '/training_set/'):
            os.mkdir(file_path + '/training_set/')

        pdb.set_trace()
        all_fov_used = np.arange(start_fov_idx, start_fov_idx + index_limit).tolist()
        for i,fov_idx_value in enumerate(all_fov_used):
            if fov_idx_value in rand_idx_list:
                imsave( file_path + '/validation_set/validation_tile_' + str(valid_count) + '.tif',all_tif_files[i])
                np.save(file_path + '/validation_set/validation_tile_' + str(valid_count) + '_seg.npy', all_npy_files[i])
                valid_count += 1
            else:
                imsave( file_path + '/training_set/training_tile_' + str(train_count) + '.tif',all_tif_files[i])
                np.save( file_path + '/training_set/training_tile_' + str(train_count) + '_seg.npy',all_npy_files[i])
                train_count += 1

        return None

