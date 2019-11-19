import pickle 
import pyvips
import os
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)



#top_dir = "/media/store/krs/caseFiles"
top_dir = "/Volumes/SD/caseFiles"

# this tool will create all cropped images


# this df contains all the case information
df_name = "process_df.pkl"
#df_path = "/Volumes/SD/caseFiles/vwi_proj"
df_file_path = os.path.join(top_dir, df_name)
df = pd.read_pickle(df_file_path)


#print(df.head())
#print(df["Image_ID"].values.dtype)

# this file_namethe case 1 crop info
# this is actual pickle, not df

file_name_crop = os.path.join(top_dir,"crop_info_study_1.pkl")
crop_info = pickle.load( open(file_name_crop , "rb" ) )

relabel_paths = True
in_dir = "vwi_proj"
out_dir = 'vwi_proc'

in_dir_path = os.path.join(top_dir, in_dir)
out_dir_top = os.path.join(top_dir, out_dir)



study_id = "1"

overwrite_files = False

if (relabel_paths == True):
    # only overwrite if all the files exist in the new location
    overwrite_on_exist = True
    for key, data in crop_info.items():
        file_split = os.path.split(data["file"])
        
        case_dir = os.path.split(file_split[0])[-1]
        new_path = os.path.join(in_dir_path, case_dir, file_split[-1])
        #print(new_path)
        crop_info[key]["file"] = new_path

        from_all_list = df[df["full_path"] == data["file"]]["full_path"].values

        if (len(from_all_list) > 0):
            
            df.loc[df["full_path"] == data["file"], "full_path"] = new_path

            #from_new = df[df["full_path"] == new_path]["full_path"].values[0]

            try:
                with open(new_path, 'rb') as f:
                    overwrite_on_exist = False
                    pass
            except IOError:
                print("File not accessible {0}".format(new_path))

        #print(from_all)
        #print(from_new)
        if ((overwrite_on_exist == True) and  (overwrite_files == True)):
            print ("warning overwriting location of the image files")
            df.to_pickle(df_file_path)
            with open(df_file_path, 'wb') as handle:
                pickle.dump(crop_info, handle)

n_pages = 9 # number of additional resolutions

# get everything that starts with
meta_keep = ["openslide", "aperio"]
other_keep = ["resolution-unit"]


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

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

sorted_crop = sorted(crop_info.items(), key = lambda x: x[1]['crop_id'])

for idx, crop_ in enumerate(sorted_crop):
    crop_[1]["crop_id"] = idx

#print (sorted_crop)
data_list = []

for crop_data in sorted_crop:
    crop_key = crop_data[0]
    data = crop_data[1]

    image_id = int(crop_key.split("_")[0])
    #print(image_id.dtype)
    data_row = df[df["Image_ID"] == image_id]
    stain = data_row["Stain"].values[0]
    specimen_id = data_row["Specimen_ID"].values[0]
    case_id = "case " + str(data_row["study"].values[0])
    slide_id = data_row["Slide_ID"].values[0]
    file_name = data_row["file_name"].values[0]
    file_path = data_row["full_path"].values[0]

    data_list.append([case_id, crop_key, specimen_id, slide_id, image_id,
        data["crop_id"], stain, data["thickness"], data["gap"], file_name, file_path])
#print(data_list)

df_test = pd.DataFrame(data_list, columns =["case_id", "slice_name",
    "specimen_id", "slice_id", "image_id", "slice_index",
    "stain", "slice_thickness", "gap", "file_name", "file_path"])
#print(df_test.head())
#df_test.to_csv( "case_1.csv")


crop_stack = []
crop_names = []
in_stack = []
mag_list = []
#mag2_list =[]
region_list = []
coordinates = []
dx = []
max_w = int(0)
max_h = int(0)
for crop_data in sorted_crop:
    crop_key = crop_data[0]
    data = crop_data[1]

    split_path = os.path.split(data['file'])
    #file_name = os.path.splitext(split_path[-1])[0]
    next_dir = os.path.split(split_path[0])[-1]
    out_dir_path = os.path.join(out_dir_top, next_dir)
    #print(out_dir_path)

    new_file_name = "case_{0}_im_{1:04d}.tiff".format(study_id, data["crop_id"])

    crop_names.append(new_file_name)
    out_file = os.path.join(out_dir_path, new_file_name)

    crop_stack.append(out_file)

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    im = pyvips.Image.new_from_file(data['file'])
    in_stack.append(data['file'])
    fields = im.get_fields()
    im_dict = {}
    for field in fields:
        im_dict[field] = im.get(field)
    
    #print(im_dict)
    region_list.append(data["region"])
    region = data["region"]
    max_w = max(max_w, region["width"])
    max_h = max(max_h, region["height"])
    #print(region)
    mag_list.append(float(im_dict['aperio.AppMag'] ))
    #mag2_list.append(float(im_dict['openslide.objective-power'] ))
    coordinates.append((im_dict['openslide.mpp-x'], im_dict['openslide.mpp-y']))

#print(mag2_list)

df_test["crop_names"] = crop_names
df_test["crop_paths"] = crop_stack
df_test["base_mag"] = mag_list


#df_test.to_csv( "case_1.csv")
#quit()
mpp_x_list = []
mpp_y_list = []
x_res_list = []
y_res_list = []
n_tiles = n_pages + 1
max_dim = max(max_w, max_h)
n_pow2 = shift_bit_length(max_dim)
shift = (n_pow2 - max_dim) // 2
#print(shift)
cur_file = ""
# could be put in the next loop if they are changing
df_test["base_width"] = n_pow2
df_test["base_height"] = n_pow2
df_test["n_tiles"] = n_tiles

#logging.basicConfig(level=logging.DEBUG)
for f_in, region, im_out in zip(in_stack, region_list, crop_stack):
    print("reading {0}".format(f_in))
    if (f_in != cur_file):
        im = pyvips.Image.new_from_file(f_in, access='sequential')
        cur_file = f_in

    fields = im.get_fields()
    im_dict = {}
    for field in fields:
        check_field = field.split(".")[0]
        if ((check_field in meta_keep )): # or
            #(check_field in other_keep )):
            im_dict[field] = [ im.get_typeof(field), im.get(field)]
    #print(im_dict)

    
    left = region["left"]
    top = region["top"]
    width = region["width"]
    height = region["height"]
    shift_x = (n_pow2 - width) // 2
    shift_y = (n_pow2 - height) // 2

    crop = im.extract_area(left, top, width, height)

    blank_color = pyvips.Image.black(n_pow2, n_pow2, bands = 1)
    
    r, g, b, a = crop.bandsplit()
    # r = crop.extract_band(0).cast("ushort")
    # g = crop.extract_band(1).cast("ushort")
    # b = crop.extract_band(2).cast("ushort")
    # convert to luminesence (maybe not the best idea, but probably ok for registration)
    # may be a bad choice for 
    gray = ((21 * r + 72 * g + 7 * b) // 100).cast("uchar")
    #gray = ((r + g + b) // 3).cast("uchar")

    blank = pyvips.Image.black(n_pow2, n_pow2, bands = 1)
    blank = gray.embed(
                    shift_x, shift_y,
                    blank.get("width"), blank.get("height"),
                    extend="VIPS_EXTEND_WHITE")
                    #extend="VIPS_EXTEND_COPY")

    invert = blank.invert()

    # np_2d = np.ndarray(buffer=invert.write_to_memory(),
    #                dtype=format_to_dtype[invert.format],
    #                shape=[invert.height, invert.width])
    # mpp_x = float(im.get("openslide.mpp-x"))
    # wgt_x = np.linspace(mpp_x/2.0, mpp_x/2.0*(n_pow2-1), n_pow2)
    # # mpp_y = float(im.get("openslide.mpp-y"))
    # # wgt_y = np.linspace(mpp_y/2.0, mpp_y/2.0*(n_pow2-1), n_pow2)
    # wgt_y = wgt_x
    # x_avg = np_2d.average(axis=0, weights=wgt_x)
    # y_avg = np_2d.average(axis=1, weights=wgt_y)
    # avg = np.average(np_2d)
    
        
    #test = indx * blank.invert()

    # indx = pyvips.Image.xyz(n_pow2, n_pow2)
    #indx.tiffsave("test3.tif", compression = "VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE")
    # x = indx.extract_band(0)
    # y = indx.extract_band(1)
    # img_avg = invert.avg()
    # x_bar = (x * invert).avg() / img_avg
    # y_bar = (y * invert).avg() / img_avg

    # print(x_bar, y_bar)
    #print(im.get("xres"), im.get("yres"))
    #convert mpp to pixels/mm
    mpp_x = float(im.get("openslide.mpp-x"))
    mpp_y = float(im.get("openslide.mpp-y"))
    mpp_x_list.append(mpp_x)
    mpp_y_list.append(mpp_y)
    x_res = 1000.0/(mpp_x)
    y_res = 1000.0/(mpp_y)
    x_res_list.append(x_res)
    y_res_list.append(y_res)
    test = invert.copy(xres = x_res,
                       yres = y_res,
                       )
    test.set_type(pyvips.GValue.gstr_type, "resolution-unit", "mm")
    for field, data in im_dict.items():
        test.set_type(data[0], field, data[1])
        #print(field, test.get(field))

    print("writing {0}".format(im_out))
    test.tiffsave(im_out, compression = "VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE",
                properties=True, strip=False,
                tile=True,
                tile_width=n_pow2 // (2**n_pages),
                tile_height=n_pow2 // (2**n_pages),
                pyramid=True)#, resunit="mm")

    # create an image of size 

    
    # crop_res = im.extract_area(region["left"], region["top"],
    #                        region["width"], region["height"])

    # #crop.tiffsave("test.tif", compression = "VIPS_FOREIGN_TIFF_COMPRESSION_LZW") 
    # crop.tiffsave("test.tif", compression = "VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE") 
    #quit()
df_test["xres"] = x_res_list
df_test["yres"] = y_res_list
df_test["mpp-x"] = mpp_x_list
df_test["mpp-y"] = mpp_y_list

df_test.to_csv(os.path.join(top_dir,  "case_1.csv"))
df_test.to_pickle(os.path.join(top_dir, "case_1.pkl"))

# need to figure out the command to embed a picture in a larger picture
# only care about shifting the first image I think
# also need to resample to lower resolution, but if I can get it into the correct form
# then maybe it will be easier to resample by some power of two.
# 

# start with extracting the bands

#print(coordinates)