import pandas as pd
import numpy as np

def prepar_tie(detected_transcript,all_Z_DAPI):
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