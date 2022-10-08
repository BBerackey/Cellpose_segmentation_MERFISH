"""
Auther: Bereket Berackey

"""
import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import dask_image.imread
# import scipy as sp
import pandas as pd
import concurrent.futures
import multiprocessing as mp
import h5py
import cell_data_generator_func
import dapi_reader
import prepare_tile
import final_cell_data_generator
import pickle
import datetime


import cv2 as cv
import skimage as sk
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from matplotlib import path
from numba import jit
import time
import pdb
import xarray as xr
import json

mpl.rcParams['figure.dpi'] = 300

import cellpose
from urllib.parse import urlparse
from cellpose import models, core
# call logger_setup to have output of cellpose written
from cellpose.io import logger_setup


start_time = time.time()

def debug_func():
    cell_by_gene = []
    cell_meta_data = []
    mp2_fun_input = []
    file_path = 'U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_2/'
    with open(file_path + 'all_z_results.pickle', 'rb') as f:
        all_results = pickle.load(f)
    all_cell_meta_data = all_results['cell_meta_data']
    all_cell_gene_data = all_results['cell_by_gene']

    import pdb
    pdb.set_trace()

    print('done loading the data')
    detected_transcript = pd.read_csv(file_path + 'detected_transcripts.csv')
    detected_transcript['fov'] = detected_transcript.fov.astype('category')
    fovs = list(detected_transcript.fov.cat.categories)
    try:
        with open(file_path + 'mp2_fun_input.pickle', 'rb') as f:
            mp2_fun_input = pickle.load(f)
        print('data already exists')
    except:
        mp2_fun_input = []
        for fov_idx, fov_selected in enumerate(fovs):
            try:
                fov_all_cell_gene_data = []
                fov_all_cell_meta_data = []
                for z_idx in range(7):
                    fov_all_cell_meta_data.append(
                        all_cell_meta_data[z_idx][all_cell_meta_data[z_idx]['fov'] == fov_selected])
                    fov_all_cell_gene_data.append(all_cell_gene_data[z_idx].loc[fov_all_cell_meta_data[z_idx].index,
                                                  :])  # filter the cell by gene based on the index of cells in the fov selected
                    # import pdb;
                    # pdb.set_trace()
                mp2_fun_input.append((fov_all_cell_gene_data, fov_all_cell_meta_data))
            except IndexError:
                print(f'error no {fov_selected} fov ')
                pass
    # mp2_pool_result = final_cell_data_generator.final_data_generator(mp2_fun_input[684])
    # print(mp2_pool_result[1])
    pool = mp.Pool(mp.cpu_count())
    mp2_pool_result = pool.map_async(final_cell_data_generator.final_data_generator,mp2_fun_input)
    mp2_all_results = mp2_pool_result.get()

    cell_by_gene = []
    cell_meta_data = []
    for mp2_result in mp2_all_results:
        if not (mp2_result[0].empty):
            cell_by_gene.append(mp2_result[0])
            cell_meta_data.append(mp2_result[1])


    cell_by_gene = pd.concat(cell_by_gene)
    cell_meta_data = pd.concat(cell_meta_data)
    # save the cell by gene and meta data
    cell_by_gene.to_csv(file_path + 'cellpose_cell_by_gene.csv')
    cell_meta_data.to_csv(
        file_path + 'cellpose_cell_meta_data.csv')


def main():
    print('this is version 2')
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d' % use_GPU)
    logger_setup()

    file_paths = ['U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_2/'] #,
                  # 'U://MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_0/'
                  # 'Z:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_1/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_2/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_0/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_1/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_2/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_3/']
    raw_data_paths = ['U:/Lab/MERFISH_Imaging data/MERFISH_Raw data/202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/']

    for f_idx,file_path in enumerate(file_paths):
        # read all the dapi images
        all_Z_DAPI = dapi_reader.parallel_dapi_reader(file_path, 7)
        raw_data_path = raw_data_paths[f_idx]

        # read the detected transcript csv file
        detected_transcript = pd.read_csv(file_path + 'detected_transcripts.csv')
        detected_transcript['fov'] = detected_transcript.fov.astype('category')
        fovs = list(detected_transcript.fov.cat.categories)

        # read in the manifest file
        f = open(file_path + 'images/manifest.json')
        manifest_file = json.load(f)
        x_start = manifest_file['bbox_microns'][0]
        x_end = manifest_file['bbox_microns'][2]
        y_start = manifest_file['bbox_microns'][1]
        y_end = manifest_file['bbox_microns'][3]
        x_pxls = manifest_file['mosaic_width_pixels']  # this is the second dimension of the numpy array
        y_pxls = manifest_file['mosaic_height_pixels']


        # prepare the xarray for each dapi image
        all_z_dapi_xarray = []
        pxl_micron = 0

        for z_idx in range(len(all_Z_DAPI)):
            dapi_img = all_Z_DAPI[z_idx]
            x_loc = np.linspace(x_start, x_end, num=x_pxls)
            y_loc = np.linspace(y_start, y_end, num=y_pxls)
            dapi_xarray = xr.DataArray(dapi_img, coords=[('y', y_loc), ('x', x_loc)])
            all_z_dapi_xarray.append(dapi_xarray)
            if z_idx == 0:
                pxl_micron = np.diff(x_loc).min()


        # # prepare tile
        x_bounds,y_bounds = prepare_tile.prepar_tile(detected_transcript,fovs) #(df_fov_arange,raw_data_path,pxl_micron)

        # Modify the x_bounds and y_bounds to ensure overlap between the fovs
        x_overlap = x_bounds[:, 0].min() - x_start
        y_overlap = y_bounds[:, 0].min() - y_start

        # only modify the xy_min bounds
        x_bounds[:, 0] = x_bounds[:, 0] - x_overlap
        y_bounds[:, 0] = y_bounds[:, 0] - y_overlap


        # prepare the tiles, performing the tile within the main function to prevent memory overload
        all_z_tiles = []
        all_z_xarray_tiles = []
        for z_idx in range(len(all_Z_DAPI)):
            y_coord = all_z_dapi_xarray[z_idx].coords['y']
            x_coord = all_z_dapi_xarray[z_idx].coords['x']

            dapi_tile_list = []
            xarray_tile_list = []
            for idx in range(x_bounds.shape[0]):
                # xy min and max in micron values
                x_min, x_max, y_min, y_max = x_bounds[idx, 0], x_bounds[idx, 1], y_bounds[idx, 0], y_bounds[idx, 1]
                temp_xarray = all_z_dapi_xarray[z_idx].loc[dict(y =y_coord[ (y_coord>=y_min) & (y_coord <= y_max)],x =x_coord[ (x_coord>=x_min) & (x_coord <= x_max)])]
                xarray_tile_list.append(temp_xarray)
                dapi_tile_list.append(temp_xarray.to_numpy())

                # * CHECK the order of x and y
            all_z_tiles.append(dapi_tile_list)
            all_z_xarray_tiles.append(xarray_tile_list)
        #with open('U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_0/cellpose debugging files/tile_xarray.pickle','wb') as f:
        #    pickle.dump([all_z_tiles,all_z_xarray_tiles],f)
        

        date_today = datetime.datetime.now()
        mm_dd_yy = str(date_today.month) + '_' + str(date_today.day) + '_' + str(date_today.year)
        try:
            with open(file_path + 'all_output_mask_' + mm_dd_yy +'.pickle', 'rb') as f:
                 z_masks = pickle.load(f)
                 print('done reading z_mask')
        except:
            # use trained cellpose model
            # model_path = 'U:/Lab/Bereket_public/Merfish_AD_project_data_analysis/cell_pose/model_AD_net/cellpose_AD_net_71122'
            model_path = 'U:/Lab/Bereket_public/Merfish_AD_project_data_analysis/cell_pose/cellpose_training_datasetall_selected_tiles/training_set/models/cellpose_AD_net_9222'
            model = models.CellposeModel(gpu=use_GPU,
                                         pretrained_model=model_path)

            diameter = 0
            chan, chan2 = 0, 0
            diameter = model.diam_labels if diameter == 0 else diameter

            flow_threshold = 0.4
            cellprob_threshold = 0
            z_masks = []
            for z_idx in range(len(all_z_tiles)):
                # run model on test images
                masks, flows, styles = model.eval(all_z_tiles[z_idx],
                                                  channels=[chan, chan2],
                                                  diameter=diameter,
                                                  flow_threshold=flow_threshold,
                                                  cellprob_threshold=cellprob_threshold)
                z_masks.append(masks)
                print(f'done preparing the mask for z layer{z_idx}')

        #     with open(file_path + 'all_output_mask_' + mm_dd_yy +'.pickle', 'wb') as f:
        #        pickle.dump(z_masks, f)
        #
        # import pdb
        # pdb.set_trace()

        # generate the cell data
        detected_transcript['gene'] = detected_transcript['gene'].astype('category')
        gene_panel = list(detected_transcript.gene.cat.categories)
        all_cell_gene_data = []
        all_cell_meta_data = []
        all_z_cell_contour = {}

        temp_all_results = {}

        for z_idx in range(len(all_z_tiles)):
            z_detected_transcript = detected_transcript[detected_transcript['global_z'] == z_idx]

            z_cell_by_gene = []
            z_cell_meta_data = []
            z_cell_contour = {}
            z_cell_contour[z_idx] = {}

            pool = mp.Pool(mp.cpu_count())
            func_input_z = []

            temp_output = []
            for fov_idx,fov_selected in enumerate(fovs):
                # since now we have overlapping fovs the

                masks = z_masks[z_idx][fov_idx]
                fov_xarray = all_z_xarray_tiles[z_idx][fov_idx]
                fov_xy = fov_xarray.x.min().to_numpy(),fov_xarray.x.max().to_numpy(),fov_xarray.y.min().to_numpy(),fov_xarray.y.max().to_numpy()


                # filter out the detected transcript dataframe based on the xy limits of the fov
                transcript_cond1 = (z_detected_transcript['global_x'] > fov_xy[0]) & (z_detected_transcript['global_x'] < fov_xy[1])
                transcript_cond2 =  (z_detected_transcript['global_y'] > fov_xy[2]) & (z_detected_transcript['global_y'] < fov_xy[3])
                fov_detected_transcript = z_detected_transcript[transcript_cond1 & transcript_cond2]

                func_input_z.append((fov_detected_transcript,masks,gene_panel,fov_selected, z_idx,fov_xy))
            pool_result = pool.map_async(cell_data_generator_func.data_generator_per_fov,func_input_z)
            all_z_results = pool_result.get()

            for result in all_z_results:
                if not result[-1]:
                    if isinstance(result[0],list): # stop if result is a list, since it is likly due to error
                        pdb.set_trace()
                    z_cell_by_gene.append(result[0])
                    z_cell_meta_data.append(result[1])
                    z_cell_contour[z_idx][list(result[2].keys())[-1]] = list(result[2].values())[-1] # collect the contours as a dictionary


            pd_func = lambda x: pd.concat(x) if len(x) > 1 else x[-1] # concatenate only if it is list of dataframes/series if not return the input first element
            all_cell_gene_data.append(pd_func(z_cell_by_gene))
            all_cell_meta_data.append(pd_func( z_cell_meta_data))
            all_z_cell_contour[z_idx] = z_cell_contour[z_idx]

            print(f'done with z layer__{z_idx}')



        all_z_results = {}
        all_z_results['cell_by_gene'] = all_cell_gene_data
        all_z_results['cell_meta_data'] = all_cell_meta_data
        all_z_results['cell_contour'] = all_z_cell_contour

        print(f'time taken...{(time.time() - start_time)}...')
        print('started generating the final result')

        with open(file_path + 'all_z_results.pickle', 'wb') as f:
            pickle.dump(all_z_results, f)

        #
        cell_by_gene = []
        cell_meta_data = []
        try:
            with open(file_path + 'mp2_fun_input.pickle', 'rb') as f:
                mp2_fun_input =  pickle.load( f)
        except:
            mp2_fun_input = []
            for fov_idx, fov_selected in enumerate(fovs):
                try:
                    fov_all_cell_gene_data = []
                    fov_all_cell_meta_data = []
                    for z_idx in range(len(all_cell_meta_data)):
                        temp_meta_data = all_cell_meta_data[z_idx][all_cell_meta_data[z_idx]['fov'] == fov_selected]
                        # check if the dataframe is empty
                        if not temp_meta_data.empty:
                            fov_all_cell_meta_data.append(temp_meta_data)
                            fov_all_cell_gene_data.append(all_cell_gene_data[z_idx].loc[temp_meta_data.index,:])  # filter the cell by gene based on the index of cells in the fov selected
                    if len(fov_all_cell_meta_data) == len(all_cell_meta_data):  # this is to ensure the fov is in all z layers
                        mp2_fun_input.append((fov_all_cell_gene_data, fov_all_cell_meta_data))
                except IndexError:
                    print(f'error no {fov_selected} fov ')
                    print(f'will skip to the next fov')
                    continue
            # with open(file_path + 'mp2_fun_input.pickle', 'wb') as f:
            #       pickle.dump(mp2_fun_input,f)


        pool = mp.Pool(mp.cpu_count())
        mp2_pool_result = pool.map_async(final_cell_data_generator.final_data_generator, mp2_fun_input)
        mp2_all_results = mp2_pool_result.get()

        cell_by_gene = []
        cell_meta_data = []

        # ensure there is a folder for saving the cell boundaries
        if not  os.path.isdir(file_path +'cellpose_cell_boundaries_optimized'):
            os.mkdir(file_path +'cellpose_cell_boundaries_optimized')

        import pdb
        pdb.set_trace()

        for mp2_result in mp2_all_results:
            if not (mp2_result[0].empty):
                cell_by_gene.append(mp2_result[0])
                cell_meta_data.append(mp2_result[1])
                current_fov = list(mp2_result[-1].keys())[0]
                h5file = h5py.File(
                    file_path + 'cellpose_cell_boundaries_optimized' + '/cell_contour_' + str(int(current_fov)) + '.h5', 'w')

                fov_xy_min_max = np.array([[x_bounds[fovs.index(current_fov), 0], y_bounds[fovs.index(current_fov), 0]],
                                           [x_bounds[fovs.index(current_fov), 1],
                                            y_bounds[fovs.index(current_fov), 1]]])
                fov_xy_bounds_bound = h5file.create_dataset('fov_bounds/xy_min_max', data=fov_xy_min_max)

                for current_cell_dict in mp2_result[-1][current_fov]:
                    current_cell_idx = list(current_cell_dict.keys())[0]
                    # add the contour of the zeroth cell index
                    if int(current_fov) in all_z_cell_contour[0].keys():  # check if that fov exists
                        z_specific_cell_contour = h5file.create_dataset(
                            'cell_indexes/' + current_cell_idx + '/' + 'zIndex_0',
                            data=np.array(all_z_cell_contour[0][int(current_fov)][
                                              current_cell_idx]))
                        # add the contour of the cells starting from z-layer =1
                        for z_i, z_idx in enumerate(current_cell_dict[current_cell_idx].keys(), 1):
                            if int(current_fov) in all_z_cell_contour[z_i].keys():  # check if that fov exists

                                try:
                                    z_specific_cell_contour = h5file.create_dataset(current_cell_idx + '/' + z_idx,
                                                                                    data=np.array(
                                                                                        all_z_cell_contour[z_i][
                                                                                            int(current_fov)][
                                                                                            current_cell_dict[
                                                                                                current_cell_idx][
                                                                                                z_idx][-1]]))
                                except:
                                    pdb.set_trace()
                                    print((z_i, current_fov, current_cell_dict[current_cell_idx][z_idx][-1]))

                h5file.close()

        cell_by_gene = pd.concat(cell_by_gene)
        cell_meta_data = pd.concat(cell_meta_data)
        # save the cell by gene and meta data
        cell_by_gene.to_csv(file_path + 'cellpose_cell_by_gene_optimized.csv')
        cell_meta_data.to_csv(
            file_path + 'cellpose_cell_meta_data_optimized.csv')


if __name__ == '__main__':
   main()
   # debug_func()

