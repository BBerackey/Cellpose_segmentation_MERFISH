import numpy as np
import cellpose
import pandas as pd
from matplotlib import path
import concurrent.futures
from numba import jit
import pickle

# numba unique functions
@jit(nopython=True)
def numba_unique(rgb_mask):
    reshaped_mask = rgb_mask.reshape(rgb_mask.shape[0] * rgb_mask.shape[1], rgb_mask.shape[2])
    unique_list = []
    whole_list = []
    all_list = []

    for idx_1 in range(reshaped_mask.shape[0]):
        whole_list.append(reshaped_mask[idx_1, :])
    un_val = whole_list[0]
    unique_list.append(un_val)
    i_count = 0   # count unique values
    for w in whole_list:
        temp_diff = un_val == w
        temp_sum = temp_diff.all()
        if ~temp_sum:
            if i_count == 0:
                un_val = w
                unique_list.append(w)
                n = len(unique_list)
                k = unique_list[0].shape[0]
                unique_array = np.zeros((n, k), dtype=np.uint16)
                for i in range(n):
                    unique_array[i] = unique_list[i]
                i_count = i_count + 1
                all_list.append(unique_array)
            else:
                un_val = w
                temp_log = unique_array == w
                log_idx = temp_log.nonzero()
                num_log = np.zeros(unique_array.shape, dtype=np.uint16)
                if log_idx[0].shape[0] > 0:
                    for idx in range(log_idx[0].shape[0]):
                        num_log[log_idx[0][idx], log_idx[1][idx]] = 1
                    log_prod = num_log[:, 0] * num_log[:, 1] * num_log[:, 2]
                    log_sums = 0
                    for row_idx in range(log_prod.shape[0]):
                        log_sums = log_sums + log_prod[row_idx]
                    if log_sums == 0.0:
                        i_count = i_count + 1
                        unique_list.append(w)
                        n = len(unique_list)
                        k = unique_list[0].shape[0]
                        unique_array = np.zeros((n, k), dtype=np.uint16)
                        for i in range(n):
                            unique_array[i] = unique_list[i]
                        all_list.append(unique_array)
                elif log_idx[0].shape[0] == 0:
                    i_count = i_count + 1
                    unique_list.append(w)
                    n = len(unique_list)
                    k = unique_list[0].shape[0]
                    unique_array = np.zeros((n, k), dtype=np.uint16)
                    for i in range(n):
                        unique_array[i] = unique_list[i]
                    all_list.append(unique_array)
    if i_count == 0: # if there is only one unique value
        n = len(unique_list)
        k = unique_list[0].shape[0]
        unique_array = np.zeros((n, k), dtype=np.uint16)
        for i in range(n):
            unique_array[i] = unique_list[i]

    return unique_array

# 2nd numba function

# numba functions
@jit(nopython=True)
def numba_fun_1(rgb_mask_copy, unique_rgb_mask):
    #     unique_rgb_mask_label = []
    final_rgb_mask = 255 * np.ones(rgb_mask_copy.shape, dtype=np.uint8)

    no_cell_flag = True
    for unique_idx in range(unique_rgb_mask.shape[0]):
        if not np.all(unique_rgb_mask[unique_idx, :] == np.array([255, 255, 255])):
            no_cell_flag = False
            tile_mask = np.ones(rgb_mask_copy.shape, dtype=np.uint8) * unique_rgb_mask[unique_idx, :]
            temp_mask = np.zeros(rgb_mask_copy.shape)
            log_idx_1 = (rgb_mask_copy == tile_mask).nonzero()
            for idx in range(len(log_idx_1[0])):
                temp_mask[log_idx_1[0][idx], log_idx_1[1][idx], log_idx_1[2][idx]] = 1

            temp_mask_2 = temp_mask[:, :, 0] * temp_mask[:, :, 1] * temp_mask[:, :, 2]
            temp_mask_3 = temp_mask_2 == 1
            temp_mask_4 = temp_mask_3.repeat(3).reshape(tile_mask.shape[0], tile_mask.shape[1], 3)
            log_idx_2 = temp_mask_4.nonzero()
            for idx in range(len(log_idx_2[0])):
                final_rgb_mask[log_idx_2[0][idx], log_idx_2[1][idx], log_idx_2[2][idx]] = unique_idx
    #                 unique_rgb_mask_label.append(unique_idx)
    #     reshaped_mask_label = np.reshape(rgb_mask_copy,[rgb_mask_copy.shape[0]*rgb_mask_copy.shape[1],rgb_mask_copy.shape[2]])
    unique_rgb_mask_label = numba_unique(final_rgb_mask)
    #     unique_rgb_mask_label = np.unique(reshaped_mask_label,axis=0) # returns array with unique value
    #     unique_rgb_mask_label.append(255)
    return final_rgb_mask, unique_rgb_mask_label,no_cell_flag  # np.array(unique_rgb_mask_label)


@jit(nopython=True)
def numba_fun_2(fov_mask, unique_rgb_mask_label, unique_idx, fov_y_min, fov_x_min):
    fov_log_idx = fov_mask == np.max(unique_rgb_mask_label[unique_idx, :])
    ind = fov_log_idx.nonzero()
    ind_1 = ind[0].reshape(ind[0].shape[0], 1)
    ind_2 = ind[1].reshape(ind[1].shape[0], 1)
    ind = np.concatenate((ind_1, ind_2), axis=1)  # [x_ind,y_ind]
    ind = np.add(ind * 0.108, np.array([fov_y_min, fov_x_min]))  # convert to pxl to microns  and the offset
    label_x_min, label_x_max = ind[:, 0].min(), ind[:, 0].max()  # these are the xy max and mins
    label_y_min, label_y_max = ind[:, 1].min(), ind[:, 1].max()

    # calculate x and y coord of cell center
    center_y = label_x_min + (label_x_max - label_x_min) / 2
    center_x = label_y_min + (label_y_max - label_y_min) / 2
    # ** note the center_x and center_y are switched inorder to match the DAPI

    # filter the detected_transcript based on the xy indexs
    cell_contour_1 = []
    cell_contour_2 = []
    for i in np.unique(ind[:, 0]):
        temp_log = ind[:, 0] == i
        log_idx = temp_log.nonzero()
        temp_ind_list = []
        if log_idx[0].shape[0] > 0:
            for idx in range(log_idx[0].shape[0]):
                temp_ind_list.append(ind[log_idx[0][idx], :])
            n = len(temp_ind_list)
            k = temp_ind_list[0].shape[0]
            temp_ind = np.zeros((n, k))
            for i in range(n):
                temp_ind[i] = temp_ind_list[i]

        #         temp_ind = ind[ind[:,0] == i]
        temp_coord_1 = temp_ind[temp_ind[:, 1] == temp_ind[:, 1].max()][0]
        temp_coord_2 = temp_ind[temp_ind[:, 1] == temp_ind[:, 1].min()][0]

        #         cell_contour_1.append(temp_ind)
        cell_contour_1.append((temp_coord_1[0], temp_coord_1[1]))
        cell_contour_2.append((temp_coord_2[0], temp_coord_2[1]))
    cell_contour = cell_contour_1 + cell_contour_2[::-1]

    return cell_contour, center_x, center_y, label_x_min, label_x_max, label_y_min, label_y_max


def data_generator_per_fov(func_input_arg):
    # fov_detected_transcript, mask, selected_dapi, gene_panel, fov_selected, z_idx = func_input_arg
    fov_detected_transcript, mask, gene_panel, fov_selected, z_idx = func_input_arg
    print('started data generation for .... fov ' + str(fov_selected))
    # list and dict to keep record of cell_by_gene and cell_meta_data of cells within the fov
    cell_by_gene = []
    cell_meta_data = []
    cell_contour_fov = {}

    # selected_dapi = selected_dapi / 65535
    rgb_mask = cellpose.plot.mask_rgb(mask)
    unique_rgb_mask = numba_unique(rgb_mask)
    rgb_mask_copy = rgb_mask.copy()
    rgb_mask_out, unique_rgb_mask_label,no_cell_flag = numba_fun_1(rgb_mask_copy, unique_rgb_mask)
    fov_mask = rgb_mask_out.max(axis=2)  # 2D mask



    # x_min,x_max,y_min,y_max of the fov
    fov_x_min, fov_x_max = fov_detected_transcript.global_x.min(), fov_detected_transcript.global_x.max()
    fov_y_min, fov_y_max = fov_detected_transcript.global_y.min(), fov_detected_transcript.global_y.max()

    # then filter the transcript for that specific z_layer
    fov_detected_transcript = fov_detected_transcript[fov_detected_transcript['global_z'] == z_idx]
    # the reason for not filtering by z-layer first is in order to approx. the real fov size

    # loop thorough each labeled pixels
    temp_cell_transcript_record = []
    cell_contour_record = {}
    uni_ind = []
    for unique_idx in range(unique_rgb_mask_label.shape[0]):
        try:
            if (not (np.max(unique_rgb_mask_label[unique_idx, :]) == 255)) & (not no_cell_flag):

                cell_ID = 'cell_in_' + 'reg_' + str(0) + '_' + str(unique_idx) + '_Z_' + str(z_idx) + '_' + str(
                    np.random.randint(99999, 9999999))
                output_tuple = numba_fun_2(fov_mask, unique_rgb_mask_label, unique_idx, fov_y_min, fov_x_min)
                cell_contour, center_x, center_y, label_x_min, label_x_max, label_y_min, label_y_max = output_tuple
                # switch the x and y coord
                cell_contour = [(c[1], c[0]) for c in cell_contour]
                cell_contour_record[cell_ID] = cell_contour
                points_to_check = fov_detected_transcript.loc[:, ['global_x', 'global_y']].to_numpy()
                polygon_temp = path.Path(cell_contour)
                bool_index = polygon_temp.contains_points(points_to_check)
                bool_df = pd.array(bool_index)
                cell_detected_transcripts = fov_detected_transcript[bool_df]
                cell_gene_count = {
                    gene: cell_detected_transcripts[cell_detected_transcripts['gene'] == gene].shape[0] for gene in
                    gene_panel}
                temp_cell_by_gene = pd.DataFrame.from_dict(cell_gene_count, orient='index', columns={cell_ID}).T
                temp_cell_by_gene = pd.concat(
                    [pd.DataFrame([[center_x, center_y]], columns=['center_x', 'center_y'], index=[cell_ID]),
                     temp_cell_by_gene], axis=1)
                cell_by_gene.append(temp_cell_by_gene)

                # cell meta data
                #                 cell_meta_data.append(pd.DataFrame(np.reshape(np.array([fov_selected,center_x,center_y,label_x_min,label_x_max,label_y_min,label_y_max,xy_max[0],xy_max[1]]),[1,9]),columns =['fov','center_x','center_y','x_min','x_max','y_min','y_max','zxy_xmax','zxy_ymax'],index = [cell_ID]))
                cell_meta_data.append(pd.DataFrame(np.reshape(np.array(
                    [fov_selected, center_x, center_y, label_x_min, label_x_max, label_y_min, label_y_max]),
                    [1, 7]),
                    columns=['fov', 'center_x', 'center_y', 'x_min', 'x_max',
                             'y_min', 'y_max'], index=[cell_ID]))
                cell_contour_fov[fov_selected] = cell_contour_record
        except ValueError:
            # import pdb;
            # pdb.set_trace()
            no_cell_flag = True
            pass
    if len(cell_by_gene) !=0:
        cell_by_gene = pd.concat(cell_by_gene)
        cell_meta_data = pd.concat(cell_meta_data)
    print('done.... fov ' + str(fov_selected))

    return cell_by_gene,cell_meta_data,cell_contour_fov,no_cell_flag




def cell_data_generator(detected_transcript,all_z_tiles,z_masks,fovs):

    detected_transcript['gene'] = detected_transcript['gene'].astype('category')
    gene_panel = list(detected_transcript.gene.cat.categories)

    all_cell_by_gene = []
    all_cell_meta_data = []
    all_z_cell_contour = {}
    # masks = list(masks)
    for z_idx in range(3):
        dapi_tile_list = all_z_tiles[z_idx]
        masks = z_masks[z_idx]
        cell_by_gene = []
        cell_meta_data = []
        xy_max = []
        cell_contour_fov = {}

        for f_i, fov_selected in enumerate(fovs):
            fov_idx = fovs.index(int(fov_selected))
            selected_dapi = dapi_tile_list[fov_idx] / 65535
            rgb_mask = cellpose.plot.mask_rgb(masks[fov_idx])
            unique_rgb_mask = numba_unique(rgb_mask)
            rgb_mask_copy = rgb_mask.copy()
            rgb_mask_out, unique_rgb_mask_label = numba_fun_1(rgb_mask_copy, unique_rgb_mask)

            fov_mask = rgb_mask_out.max(axis=2)  # 2D mask
            fov_detected_transcript = detected_transcript[detected_transcript['fov'] == int(fov_selected)]

            # x_min,x_max,y_min,y_max of the fov
            fov_x_min, fov_x_max = fov_detected_transcript.global_x.min(), fov_detected_transcript.global_x.max()
            fov_y_min, fov_y_max = fov_detected_transcript.global_y.min(), fov_detected_transcript.global_y.max()

            # then filter the transcript for that specific z_layer
            fov_detected_transcript = fov_detected_transcript[fov_detected_transcript['global_z'] == z_idx]
            # the reason for not filtering by z-layer first is in order to approx. the real fov size

            # loop thorough each labeled pixels
            temp_cell_transcript_record = []
            cell_contour_record = {}
            uni_ind = []
            for unique_idx in range(unique_rgb_mask_label.shape[0]):
                if ~(np.max(unique_rgb_mask_label[unique_idx, :]) == 255):
                    cell_ID = 'cell_in_' + 'reg_' + str(0) + '_' + str(unique_idx) + '_Z_' + str(z_idx) + '_' + str(
                        np.random.randint(99999, 9999999))
                    output_tuple = numba_fun_2(fov_mask, unique_rgb_mask_label, unique_idx, fov_y_min, fov_x_min)
                    cell_contour, center_x, center_y, label_x_min, label_x_max, label_y_min, label_y_max = output_tuple
                    # switch the x and y coord
                    cell_contour = [(c[1], c[0]) for c in cell_contour]
                    cell_contour_record[cell_ID] = cell_contour
                    points_to_check = fov_detected_transcript.loc[:, ['global_x', 'global_y']].to_numpy()
                    polygon_temp = path.Path(cell_contour)
                    bool_index = polygon_temp.contains_points(points_to_check)
                    bool_df = pd.array(bool_index)
                    cell_detected_transcripts = fov_detected_transcript[bool_df]
                    cell_gene_count = {
                        gene: cell_detected_transcripts[cell_detected_transcripts['gene'] == gene].shape[0] for gene in
                        gene_panel}
                    temp_cell_by_gene = pd.DataFrame.from_dict(cell_gene_count, orient='index', columns={cell_ID}).T
                    temp_cell_by_gene = pd.concat(
                        [pd.DataFrame([[center_x, center_y]], columns=['center_x', 'center_y'], index=[cell_ID]),
                         temp_cell_by_gene], axis=1)
                    cell_by_gene.append(temp_cell_by_gene)

                    # cell meta data
                    #                 cell_meta_data.append(pd.DataFrame(np.reshape(np.array([fov_selected,center_x,center_y,label_x_min,label_x_max,label_y_min,label_y_max,xy_max[0],xy_max[1]]),[1,9]),columns =['fov','center_x','center_y','x_min','x_max','y_min','y_max','zxy_xmax','zxy_ymax'],index = [cell_ID]))
                    cell_meta_data.append(pd.DataFrame(np.reshape(np.array(
                        [fov_selected, center_x, center_y, label_x_min, label_x_max, label_y_min, label_y_max]),
                                                                  [1, 7]),
                                                       columns=['fov', 'center_x', 'center_y', 'x_min', 'x_max',
                                                                'y_min', 'y_max'], index=[cell_ID]))
                    cell_contour_fov[fov_selected] = cell_contour_record
            print('done.... fov ' + str(fov_selected))
        all_z_cell_contour[z_idx] = cell_contour_fov
        cell_by_gene = pd.concat(cell_by_gene)
        cell_meta_data = pd.concat(cell_meta_data)

        all_cell_by_gene.append(cell_by_gene)
        all_cell_meta_data.append(cell_meta_data)

        print('done ..... z layer' + str(z_idx))
    return all_cell_by_gene,all_cell_meta_data



