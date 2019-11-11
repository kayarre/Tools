import pickle 
#import pyvips
import os
import pandas as pd
import numpy as np
import tifffile as tiff
import SimpleITK as sitk


import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

from read_vips import parse_vips

# this register the cropped images

#df_path = "/Volumes/SD/caseFiles/vwi_proj/process_df.pkl"
#crop_dir = '/Volumes/SD/caseFiles/vwi_proc'
#csv = pd.read_csv(os.path.join(crop_dir,  "case_1.csv"))
#df = pd.read_pickle(os.path.join(crop_dir, "case_1.pkl"))

case_file = "case_1.pkl"
top_dir = "/media/store/krs/caseFiles"

df = pd.read_pickle(os.path.join(top_dir, "case_1.pkl"))

#print(df.head())

relabel_paths = True
in_dir = "vwi_proj"
out_dir = 'vwi_proc'
print(df.head())
print(df.columns)
#print(df["Image_ID"].values.dtype)

#study_id = "1"


format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

def get_additional_info(pd_row):
    pd_data = pd_row[1]
    ff_path = pd_data["crop_paths"]
    ff = tiff.TiffFile(ff_path)

    base_res_x = pd_data["mpp-x"]
    base_res_y = pd_data["mpp-y"]

    print(base_res_x, base_res_y)
    #meta_data_orig = parse_vips(ff.pages[0].description)

    base_shape = ff.pages[0].shape
    pages = []
    for page in ff.pages:
        x_size = page.imagewidth
        y_size = page.imagelength
        xscale = base_shape[0] // x_size
        yscale = base_shape[1] // y_size
        pages.append(dict(size_x=x_size, size_y=y_size, scale_x = xscale, scale_y=yscale))

    return pages

# n_max is the maxmimum picture size for registration
def stage_1_transform(reg_dict, n_max):
    #print(fixed[1]["crop_paths"])
    ff_path = reg_dict["f_row"]["crop_paths"]
    ff = tiff.TiffFile(ff_path)

    tf_path = reg_dict["t_row"]["crop_paths"]
    tf = tiff.TiffFile(tf_path)

    
    #print(stuff)
    #stuff = {}
    #return stuff

# this is the registration loop
tmp_index = 0
reg_n = []
for row in df.iterrows():

    if (tmp_index == row[0]):
        f_r = row # the data
        f_pg_info = get_additional_info(f_r)
        continue
    else:
        t_r = row
        t_pg_info = get_additional_info(t_r)

    # leave the index for now
    reg_n.append(dict(f_row = f_r[1], t_row = t_r[1], f_page=f_pg_info, t_page=t_pg_info))

    #print(template[0], fixed[0])

    stage_1_params = sitk.GetDefaultParameterMap("translation")
    stage_1_params["NumberOfResolutions"] = ['1']
    print(sitk.PrintParameterMap(stage_1_params))
    quit()
    stage_1_transform(reg_n[-1], N_max=256)
    


    f_r = t_r
    f_pg_info = t_pg_info
    #print(template_index, fixed_index)
    tmp_index += 1
    
