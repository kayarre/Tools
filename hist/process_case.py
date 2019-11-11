import pickle 
import pyvips
import os
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

# this register the cropped images

#df_path = "/Volumes/SD/caseFiles/vwi_proj/process_df.pkl"
crop_dir = '/Volumes/SD/caseFiles/vwi_proc'
#csv = pd.read_csv(os.path.join(crop_dir,  "case_1.csv"))
df = pd.read_pickle(os.path.join(crop_dir, "case_1.pkl"))

print(df.head())
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

# this is the registration loop
tmp_index = 0
for row in df.iterrows():

    if (tmp_index == row[0]):
        fixed = row # the data
        continue
    else:
        template = row

    print(template[0], fixed[0])


    #trans = get_transform(fixed, template)



    fixed = template
    #print(template_index, fixed_index)
    tmp_index += 1
    
