import numpy as np
import pandas as pd
import numba
from numba.typed import List
import pdb

@numba.jit(nopython=True)
def numba_func_cell_merge(numpy_cell_by_gene,numpy_cell_meta_data,cell_idx_list,all_col_index):
    all_numba_output = []
    center_x_col, center_y_col, x_max_col, x_min_col, y_max_col, y_min_col = all_col_index
    for idx in range(len(cell_idx_list)):
        cell_idx = cell_idx_list[idx]
        center_datatype = type(numpy_cell_meta_data[0][idx,x_max_col])
        xy_max = np.zeros((7, 2), dtype=center_datatype)
        xy_min = np.zeros((7, 2),dtype = center_datatype)

        # if isinstance(idx,list): # ***note isinstance is not supported in numba
        #     raise Exception('error: cell index duplicated')
        # take the row of the z = 0  and compare it to all other layers
        center_x = numpy_cell_by_gene[0][idx,center_x_col]
        center_y = numpy_cell_by_gene[0][idx, center_y_col]

        csum_gene_array = numpy_cell_by_gene[0][idx,2:]
        xy_max[0, 0] = numpy_cell_meta_data[0][idx, x_max_col]
        xy_max[0, 1] = numpy_cell_meta_data[0][idx, y_max_col]

        xy_min[0, 0] = numpy_cell_meta_data[0][idx,x_min_col]
        xy_min[0, 1] = numpy_cell_meta_data[0][idx,y_min_col]
    #
        for z_idx in range(1, 7):
            all_centerx = numpy_cell_by_gene[z_idx][:,center_x_col]
            all_centery = numpy_cell_by_gene[z_idx][:,center_y_col]
            diff1 = (all_centerx - (center_x * np.ones(all_centerx.shape)))
            diff1[diff1<0] = -1*diff1[diff1<0] # taking the absolute value
            diff1_min = diff1.min()
            diff1 = diff1 - (diff1_min * np.ones(diff1.shape))
            cond1 = diff1.astype('int') == 0

            diff2 = (all_centery - (center_y * np.ones(all_centery.shape)))
            diff2[diff2 < 0] = -1 * diff2[diff2 < 0] # taking the absolute value
            diff2_min = diff2.min()
            diff2 = diff2 - (diff2_min * np.ones(diff2.shape))
            cond2 = diff2.astype('int') == 0

            cond = cond1 & cond2

    #         # turn into absolute value

            # apply the condition, numba does not support 2D logical indexing
            if cond.any(): # if there is atleast one True, i.e at least one aligning cell
                        log_cell_idx = cond.nonzero()
                        temp_xy_min = np.zeros((1,2))  #this is incase there are more than one aligning cells
                        temp_xy_max = np.zeros((1, 2)) # this is incase there are more than one aligning cells

                        for log_idx in range(log_cell_idx[0].shape[0]):
                            csum_gene_array = csum_gene_array + numpy_cell_by_gene[z_idx][log_cell_idx[0][log_idx],2:]

                            temp_xy_max[0,0] = np.max(np.array([temp_xy_max[0,0],numpy_cell_meta_data[z_idx][log_cell_idx[0][log_idx],x_max_col]]))
                            temp_xy_max[0,1] = np.max(np.array([temp_xy_max[0,1],numpy_cell_meta_data[z_idx][log_cell_idx[0][log_idx],y_max_col]]))

                            temp_xy_min[0, 0] = np.max(np.array([temp_xy_min[0,0],numpy_cell_meta_data[z_idx][log_cell_idx[0][log_idx],x_min_col]]))
                            temp_xy_min[0, 1] = np.max(np.array([temp_xy_min[0,1],numpy_cell_meta_data[z_idx][log_cell_idx[0][log_idx],y_min_col]]))
                        xy_min[z_idx,0] = temp_xy_min[0,0]
                        xy_min[z_idx, 1] = temp_xy_min[0, 1]

                        xy_max[z_idx,0] = temp_xy_max[0,0]
                        xy_max[z_idx, 1] = temp_xy_max[0, 1]
            elif z_idx == 1:
                break # if no cell algins to the next first z-layer, i.e. z_idx =1
                         # we do not expect cells in the other layer to align, so skip this
                         # loop to save computation time
        all_numba_output.append((csum_gene_array, cell_idx, xy_max, xy_min))
    return all_numba_output


def final_data_generator(func_input):
    all_cell_gene_data, all_cell_meta_data = func_input
    if (all_cell_gene_data[0].empty) | (all_cell_meta_data[0].empty):
        print('empty data frame found')
        return pd.DataFrame([]),pd.DataFrame([])
    else:

            gene_panel = list(all_cell_gene_data[0].columns)[2:]
            hight_value = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

            temp_cell_idx_list = list(all_cell_gene_data[0].index)
            # cell_idx_list = list(all_cell_gene_data[0].index)

            # create a typed list to make it compatible with numba
            cell_idx_list = List()
            [cell_idx_list.append(cell_idx) for cell_idx in temp_cell_idx_list]

            meta_column_list = list(all_cell_meta_data[0].columns)
            gene_column_list = list(all_cell_gene_data[0].columns)

            # specifiy the column index manually to masks it compatible with numba

            center_x_col = gene_column_list.index('center_x')
            center_y_col = gene_column_list.index('center_y')
            x_max_col = meta_column_list.index('x_max')
            x_min_col = meta_column_list.index('x_min')

            y_max_col = meta_column_list.index('y_max')
            y_min_col = meta_column_list.index('y_min')

            all_col_idx = (center_x_col, center_y_col, x_max_col, x_min_col, y_max_col, y_min_col)

   # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   #   This for loop will # change all the dataframe into numpy. It will check if the list other than the 0th layer are
   #   empty. if so,will fill them with zeros
   #   Also, it will convert them to C_contiguous type so that numba typedlist will work smoothly
            numpy_cell_by_gene = List()
            numpy_cell_meta_data = List()
            for z_idx in range(len(all_cell_gene_data)):
                loop_temp_meta= all_cell_meta_data[z_idx].values
                loop_temp_gene = all_cell_gene_data[z_idx].values
                if loop_temp_meta.size == 0:
                    loop_temp_meta = np.zeros((1,loop_temp_meta.shape[1]),dtype = loop_temp_meta.dtype)
                    loop_temp_gene = np.zeros((1,loop_temp_gene.shape[1]),dtype = loop_temp_gene.dtype)
                elif (not (loop_temp_meta.flags['C_CONTIGUOUS'])) | (not (loop_temp_gene.flags['C_CONTIGUOUS'])): # check if the array layout is C-contiguous or Fortrun
                    loop_temp_meta = np.ascontiguousarray(loop_temp_meta , dtype=loop_temp_meta.dtype)
                    loop_temp_gene = np.ascontiguousarray(loop_temp_gene, dtype=loop_temp_gene.dtype)

                numpy_cell_by_gene.append(loop_temp_gene)
                numpy_cell_meta_data.append(loop_temp_meta)


            # numpy_cell_meta_data = List(temp_numpy_cell_meta_data)
            # numpy_cell_by_gene = List(temp_numpy_cell_by_gene)
            fov_processed = all_cell_meta_data[0]['fov'].max()
            all_numba_outputs = numba_func_cell_merge(numpy_cell_by_gene,numpy_cell_meta_data,cell_idx_list,all_col_idx)

            final_cell_by_gene = []
            final_cell_meta_data = []
            for numba_output in all_numba_outputs:
                csum_gene_array,idx,xy_max,xy_min = numba_output
                csum_gene_array = np.reshape(csum_gene_array, [1, len(gene_panel)])
                final_cell_by_gene.append(pd.DataFrame(csum_gene_array, index=[idx], columns=gene_panel))

                # calculate valume and generate final cell_meta_data
                stop_z = (lambda x: x[0] if len(x) > 0 else 'NaN')(np.nonzero(xy_min.max(axis=1) == 0)[0])
                if stop_z == 'NaN':
                    stop_z = xy_min.shape[0] - 1
                width_length = xy_max - xy_min
                volume = [1 / 2 * hight_value[z_idx] * (width_length[z_idx, 0] + width_length[z_idx + 1, 0]) * np.maximum(
                    width_length[z_idx, 1], width_length[z_idx + 1, 1]) for z_idx in range(stop_z)]
                temp_cell_data = all_cell_meta_data[0].loc[idx, ['fov', 'center_x', 'center_y', 'x_min', 'x_max', 'y_min', 'y_max']]
                temp_cell_data['volume'] = sum(volume)
                temp_cell_data['x_min'] = xy_min[:stop_z, :].min(axis=0)[0]
                temp_cell_data['y_min'] = xy_min[:stop_z, :].min(axis=0)[1]
                temp_cell_data['x_max'] = xy_max.max(axis=0)[0]
                temp_cell_data['y_max'] = xy_max.max(axis=0)[1]
                final_cell_meta_data.append(temp_cell_data.to_frame().T)

            final_cell_by_gene = pd.concat(final_cell_by_gene)
            final_cell_meta_data = pd.concat(final_cell_meta_data)
            # pdb.set_trace()

            print(f'Done generating cell data for fov: {fov_processed}')
            return final_cell_by_gene, final_cell_meta_data

