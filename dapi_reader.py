import numpy as np
import dask_image.imread
import concurrent.futures


def dapi_reader(reader_func_input):
    # example file_path = 'U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_0/'
    file_path,z_idx = reader_func_input
    dapi_dask = dask_image.imread.imread(file_path + 'images/mosaic_DAPI'+'_z'+str(z_idx)+'.tif')
    # dapi_dask = dask_image.imread.imread('U:/Lab/MERFISH_Imaging data/Kim2_202112171955_12172021TREM2-5x12Mo300GP_VMSC00101/region_0/images/mosaic_DAPI'+'_z'+str(z_idx)+'.tif')
    dapi_imag = dapi_dask.compute()[0,:,:]
    print(f'done reading layer {z_idx}')
    return dapi_imag

def parallel_dapi_reader(file_path):
    all_Z_DAPI = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        reader_func_input = [(file_path,z_idx) for z_idx in range(2)]
        results = executor.map(dapi_reader,reader_func_input)
        for f in results:
            all_Z_DAPI.append(f)

    return all_Z_DAPI