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
# import h5py
import cell_data_generator_func
import dapi_reader
import prepare_tile
import final_cell_data_generator
import pickle


import cv2 as cv
import skimage as sk
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from matplotlib import path
from numba import jit
import time
import pdb

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
    file_path = 'U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_0/'
    with open(file_path + 'all_z_results.pickle', 'rb') as f:
        all_results = pickle.load(f)
    all_cell_meta_data = all_results['cell_meta_data']
    all_cell_gene_data = all_results['cell_by_gene']
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

    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d' % use_GPU)
    logger_setup()

    file_paths = ['U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_1/'] #,
                  # 'U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_0/'
                  # 'Z:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_1/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_2/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_0/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_1/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_2/',
                  # 'Z:/Lab/MERFISH_Imaging data/Kim_202201111602_20220111-WT-5xFAD10518pHip300GP_VMSC00101/region_3/']

    for file_path in file_paths:
        # read all the dapi images
        all_Z_DAPI = dapi_reader.parallel_dapi_reader(file_path)

        # read the detected transcript csv file
        detected_transcript = pd.read_csv(
            file_path + 'detected_transcripts.csv')

        # # prepare tile
        all_z_tiles, fovs = prepare_tile.prepar_tile(detected_transcript, all_Z_DAPI)

        try:
            with open(file_path + 'all_output_mask.pickle', 'rb') as f:
                 z_masks = pickle.load(f)
                 print('done reading z_mask')
        except:
            # use trained cellpose model
            model_path = 'U:/Lab/Bereket_public/Merfish_AD_project_data_analysis/cell_pose/model_AD_net/cellpose_AD_net_71122'
            model = models.CellposeModel(gpu=use_GPU,
                                         pretrained_model=model_path)

            diameter = 0
            chan, chan2 = 0, 0
            diameter = model.diam_labels if diameter == 0 else diameter

            flow_threshold = 0.4
            cellprob_threshold = 0
            z_masks = []
            for z_idx in range(7):
                # run model on test images
                masks, flows, styles = model.eval(all_z_tiles[z_idx],
                                                  channels=[chan, chan2],
                                                  diameter=diameter,
                                                  flow_threshold=flow_threshold,
                                                  cellprob_threshold=cellprob_threshold)
                z_masks.append(masks)
                print(f'done preparing the mask for z layer{z_idx}')

            with open(file_path + 'all_output_mask.pickle', 'wb') as f:
                pickle.dump(z_masks, f)

        # generate the cell data
        detected_transcript['gene'] = detected_transcript['gene'].astype('category')
        gene_panel = list(detected_transcript.gene.cat.categories)
        all_cell_gene_data = []
        all_cell_meta_data = []
        all_z_cell_contour = {}

        temp_all_results = {}

        for z_idx in range(7):
            z_detected_transcript = detected_transcript[detected_transcript['global_z'] == z_idx]

            z_cell_by_gene = []
            z_cell_meta_data = []
            z_cell_contour = {}

            pool = mp.Pool(mp.cpu_count())
            func_input_z = []

            for fov_idx,fov_selected in enumerate(fovs):
                fov_detected_transcript = z_detected_transcript[z_detected_transcript['fov'] == fov_selected]
                masks = z_masks[z_idx][fov_idx]
                func_input_z.append((fov_detected_transcript,masks,gene_panel,fov_selected, z_idx))
            pool_result = pool.map_async(cell_data_generator_func.data_generator_per_fov,func_input_z)
            all_z_results = pool_result.get()


            for result in all_z_results:
                if not result[-1]:
                    z_cell_by_gene.append(result[0])
                    z_cell_meta_data.append(result[1])
                    if z_idx in z_cell_contour:
                      z_cell_contour[z_idx].append(result[2])
                    else:
                      z_cell_contour[z_idx] = [result[2]]
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

        # with open(file_path + 'all_z_results.pickle', 'wb') as f:
        #     pickle.dump(all_z_results, f)


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
                    print(f'will skip to the next fov')
                    continue
            # with open(file_path + 'mp2_fun_input.pickle', 'wb') as f:
            #       pickle.dump(mp2_fun_input,f)

        pool = mp.Pool(mp.cpu_count())
        mp2_pool_result = pool.map_async(final_cell_data_generator.final_data_generator, mp2_fun_input)
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


if __name__ == '__main__':
   main()
   # debug_func()

