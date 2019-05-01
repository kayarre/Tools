#!/usr/bin/env python
'''Reads NeXus HDF5 files using h5py and prints the contents'''
import os 
import glob
import h5py    # HDF5 support

search_name = "*.hdf5"
dir_path = "/suppscr/fluids/sansomk/ultrasound/post_proc"
sol_files = glob.glob(os.path.join(dir_path, search_name))

for file_n in sol_files:
  print(file_n)
  f = h5py.File(file_n,  "r")
  for item in f.attrs.keys():
      print("{0:s}:{1:s}".format(item, f.attrs[item]))

  for item in f.keys():
      print("items : {0:s}".format(item))
      for item2 in f[item].attrs.keys():
          test = f[item].attrs[item2]
          #print(test.dtype)
          print("{0:s}:{1:s}".format(item2, str(f[item].attrs[item2])))
      for item3 in f[item].keys():
          print("{0:s}:{1:s}".format(item3, str(f[item][item3].shape)))
  f.close()
