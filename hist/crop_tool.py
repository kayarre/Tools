#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
from string import ascii_lowercase
import large_image

import matplotlib
matplotlib.interactive(True)
# gui_env = ['Qt5Agg', 'TKAgg','GTKAgg','Qt4Agg','WXAgg', 'MacOSX']
# for gui in gui_env:
#     try:
#         print("testing", gui)
#         matplotlib.use(gui,warn=False, force=True)
#         from matplotlib import pyplot as plt
#         break
#     except:
#         continue
# print("Using:",matplotlib.get_backend())

import matplotlib.pyplot as plt


from crop_class import single_crop, crop_data

format_to_dtype = {
    'uchar' : np.uint8,
    'char' : np.int8,
    'ushort' : np.uint16,
    'short' : np.int16,
    'uint' : np.uint32,
    'int' : np.int32,
    'float' : np.float32,
    'double' : np.float64,
    'complex' : np.complex64,
    'dpcomplex' : np.complex128,
}


#proj_dir = "/Volumes/muffins/vwi_proj"
proj_dir = "/Volumes/SD/caseFiles/vwi_proj"
#csv_file ="/Users/sansomk/code/ipythonNotebooks/VWI_proj/SpectrumData_test.csv"
csv_file ="/Users/sansomk/code/ipythonNotebooks/VWI_proj/SpectrumData_test.csv"
sub_dir ="case"
file_ext=".svs"

df = pd.read_csv(csv_file)


im_id = df[["study", "Image_ID"]]

file_name = []
full_path = []

for i in df[["study","Image_ID"]].iterrows():
    
    #print(i)
    path_test = os.path.join(proj_dir, "{0}{1:02d}".format(sub_dir,i[1]["study"]), 
                            "{0}{1}".format(i[1]["Image_ID"],file_ext))
    if (os.path.exists(path_test)):
        split_p = os.path.split(path_test)
        file_name.append(split_p[-1])
        full_path.append(path_test)
    else:
        print(" some kind of error for this file: {0}".format(path_test))


df["file_name"] = file_name
df["full_path"] = full_path

print(df.head())
df.to_csv("histology_paths.csv")


path_list = list(df[df["study"] == 1]["full_path"])
image_path = path_list[0]
base_label = os.path.splitext(os.path.split(image_path)[-1])[0]
image = large_image.getTileSource(image_path)
#image.getMetadata()
#image.getMagnificationForLevel(level=7)


label_a = base_label+"_a"
crop_lists = crop_data()
crop_lists.add_row(label_a, [1200,5000,15000, 15000, 4.0])
crop_lists.add_orig_path(label_a, image_path)

label_b = base_label+"_b"
crop_lists.add_row(label_b, [48500, 1400, 15000, 15000, 4.0])
crop_lists.add_orig_path(label_b, image_path)

label_c = base_label+"_c"
crop_lists.add_row(label_c, [92500, 2500, 15000, 15000, 4.0])
crop_lists.add_orig_path(label_c, image_path)


m_roi, err = image.getRegion(scale=crop_lists.label[label_a].scale,
                             format=crop_lists.label[label_a].format,
                             region=crop_lists.label[label_a].region
                            )

plt.imshow(m_roi)


# get a thumbnail no larger than 1024x1024 pixels
thumbnail, mimeType = image.getThumbnail( width=2048, height=1024, format='numpy')

plt.imshow(thumbnail)

study_init = 0
for image_row in df.iterrows():
    row_data = image_row[1]
    image_path = row_data["full_path"]
    if (study_init != row_data["study"]):
        study_init = row_data["study"]
        print("case {0}".format(study_init))

    image = large_image.getTileSource(image_path)
    base_label = os.path.splitext(os.path.split(image_path)[-1])[0]
    thumbnail, mimeType = image.getThumbnail( width=2048, height=1024, format='numpy')
    print(base_label, image.getMetadata())
    plt.imshow(thumbnail)
    n_crops = input("input s for skip or integer for number of crops: ")
    print(n_crops)
    if( n_crops == 's'):
        continue # skip this image
    elif ( int(n_crops) > 0):
        ascii_list = list(ascii_lowercase)
        for crop in range(int(n_crops)):
            crop_label = base_label + "_" + ascii_list.pop(0)
            print("crop {0}".format(crop))
            while True:
                init_guess = input("crop values left top width height (magnification): ")
                guess = [ int(a) for a in ' '.split(init_guess)]
                if (len(guess) < 5):
                    guess.append(4.0)
                if (len(guess) == 5):
                    break
            init_crop = single_crop(guess)
            while True:
                m_roi, err = image.getRegion(scale=init_crop.scale,
                             format=init_crop.format,
                             region=init_crop.region
                            )
                plt.imshow(m_roi)

                happy = input("are you happy? (y/n): ")
                if (happy.lower() == "y"):
                    break
                else:
                    while True:
                        update_values = input("add/subtract left top width height: ")
                        new_guess = [ int(a) for a in ' '.split(update_values)]
                        if (len(new_guess) > 1 and len(new_guess) < 5):
                            init_crop.shift_region(new_guess)
                            break
            

            crop_lists.add_row(crop_label, init_crop)
            crop_lists.add_orig_path(crop_label, image_path)

            break
        break
        
