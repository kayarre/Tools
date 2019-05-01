# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import sys
import re
import glob
import copy as cp
import h5py
import pandas as pd
import argparse

def getArgs():
  # Define the command-line arguments
  parser = argparse.ArgumentParser(
              description="A utility for converting fluent ascii files \
                          to a single HDF5 file."
                                  )

  parser.add_argument('--dir_path',
                      type=str,
                      help='The location of ascii files.',
                      required=True)
  parser.add_argument('--output_name',
                      type=str,
                      help='The name of the hdf5 file to store',
                      required=True)
  parser.add_argument('--search_name',
                      type=str,
                      help='The search name of the ascii files',
                      required=True)
  parser.add_argument('--n_steps',
                      type=int,
                      help='The number of files to store.',
                      required=True)

  return parser.parse_args()

def tryint(s):
    try:
        return int(s)
    except:
        return s
     
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    
def get_var_dict(f_vars):
  """Create a vairable dictionary to abstract writing a HDF5 file.
  Savs the points into a HDF5 file. The points will be transformed
  into 1d arrays. The resulting dataset is called points and lies in
  the root of the file.
  Parameters
  ----------
  @param f_vars 
    list of the variables from the ascii file
  @return var_dict
    dictioanry to map generic names to the variables from the file
  """
  var_dict = {}
  for var in f_vars:
    if (var == "nodenumber"):
      var_dict["node"] = var
    elif( var == "x-coordinate"):
      var_dict["X"] = var
    elif( var == "y-coordinate"):
      var_dict["Y"] = var
    elif( var == "z-coordinate"):
      var_dict["Z"] = var
    elif( var == "x-velocity"):
      var_dict["U"] = var
    elif( var == "y-velocity"):
      var_dict["V"] = var
    elif( var == "z-velocity"):
      var_dict["W"] = var
    elif( var == "absolute-pressure"):
      var_dict["P"] = var
  return var_dict

def write_fields(dbFile, time_idx, time_pt,
                 velocity, compression=None, add_fields=None):

  timeIdxGroup = dbFile.create_group("field_{0:d}".format(time_idx))
  timeIdxGroup.attrs["time"] = np.float32(time_pt)
  timeIdxGroup.create_dataset("velocity", data=velocity,
                               dtype=np.float32, compression=compression)

  if(add_fields != None):
    for fld in add_fields.keys():
      timeIdxGroup.create_dataset(fld, data=add_fields[fld],
                                  dtype=np.float32, compression=compression)


def initialize_write(dbFile, time_idx, time_pt,
                     coords, velocity, compression=None, nodes=None,
                     add_fields=None):
  """Write the velocity field into an HDF5 file.
    Will also write the corresponding time value.
    Parameters
  ---------
  hdf5File : h5py.File
      The the HDF5 file.
  time_pt : float
      The value of time associated with the written
      velocity field.
  coords : list of 1d ndarray
      A list 1d ndarray containing the points.
  velocity : list of 1d ndarray
      A list of ndarrays to write to velocity field.
  """

  pointGroup = dbFile.create_group("coordinates")
  pointGroup.attrs["nPoints"] = np.int64(coords.shape[0])
  pointGroup.attrs["dimension"] = np.int32(coords.shape[1])
  pointGroup.create_dataset("coordinates", data=coords,
                            dtype=np.float32, compression=compression)
  if(nodes != None):
    pointGroup.create_dataset("nodes", data=nodes,
                              dtype=np.int32, compression=compression)
  
  write_fields(dbFile, time_idx, time_pt,
               velocity, add_fields=None, compression=compression)


def write_time(dbFile, time, time_idx, compression=None):
  timeGroup = dbFile.create_group("time")
  timeGroup.attrs["tPoints"] = np.int32(time.shape[0])
  timeGroup.create_dataset("time", data=time,
                              dtype=np.float32, compression=compression)
  timeGroup.create_dataset("time_index", data=time_idx,
                           dtype=np.int32, compression=compression)


def create_hdf5_file(dbFile, time=None, time_idx=None, coords=None, velocity=None, fields=None):
  """Write the velocity field into an HDF5 file.
    Will also write the corresponding time value.
    Parameters
  ---------
  hdf5File : h5py.File
      The the HDF5 file.
  time_pt : float
      The value of time associated with the written
      velocity field.
  nodes: ndarray
      A 1d ndarray containing the node count
  points : list of 1d ndarray
      A list 1d ndarray containing the points.
  velocity : list of 1d ndarray
      A list of ndarrays to write to velocity field.
  fields : list of 1d ndarrays
      list of arrays with field data, e.g. pressure
  var_name_dict: dict
      creates mapping between variable names and generic names
      in case the variable names change in the future
  """

  if(time != None):
   dbFile.attrs["time_pts"] = time.shape[0]
   timeGroup = dbFile.create_group("time")
   timeGroup.create_dataset("time", data=time,
                              dtype=np.float32, compression="lzf")

  if(time_idx != None):
    timeGroup.create_dataset("time_index",
                              data=nodes, dtype=np.int32)

  if(coords != None):
    pointGroup = dbFile.create_group("coordinates")
    dbFile.attrs["nPoints"] = coords.shape[0]
    dbFile.attrs["dimension"] = coords.shape[1]
    pointGroup.create_dataset("coordinates", data=coords,
                              dtype=np.float32, compression="lzf")

  if(velocity != None):
    dbFile.attrs["nPoints"] = velocity[0].shape[0]
    vectorGroup = dbFile.create_group("vectors")
    velocityGroup = vectorGroup.create_group("velocity")

    velocityGroup.create_dataset(var_name_dict["U"], data=velocity[0],
                                 dtype=np.float32, compression="lzf")
    velocityGroup.create_dataset(var_name_dict["V"], data=velocity[1],
                                 dtype=np.float32, compression="lzf")
    velocityGroup.create_dataset(var_name_dict["W"], data=velocity[2],
                                 dtype=np.float32, compression="lzf")

  if(fields != None):
    dbFile.attrs["nPoints"] = fields[0].shape[0]
    fieldGroup = dbFile.create_group("fields")
    #fieldGroup.create_dataset(var_name_dict["P"],
    #                          data=fields[0], dtype=np.float32)


def write_binary(dbFile, sol_files, stop_n, compression=None):
  time = []
  var_dict = {}
  
  for file_idx, file_ in enumerate(sol_files):
    if (file_idx >= stop_n):
      break

    if os.path.isfile(file_):
      print(os.path.split(file_)[-1])
      split_name = file_.split('-')
      time_pt = float(split_name[-1])
      print(time_pt)
      time.append(time_pt)

      #reader = open(file_, 'rU')
      #line = reader.readline()
      #f_vars = line.split()
      #var_dict = get_var_dict(f_vars) #assume vars don't change
      #reader.close()
      #print(file_)
      data = pd.read_csv(file_, sep='\s+', usecols=np.arange(1,7)).values
      #nodes = data[:,0]
      if(file_idx == 0):
        coords = data[:,0:3]
        v = data[:,3:6]
        #p = data[:,8]
        initialize_write(dbFile, file_idx, split_name[-1], coords, v, compression)
      else:
        #p = data[:,8]
        v = data[:,3:6]
        write_fields(dbFile, file_idx, split_name[-1], v, compression)
    
  time = np.asarray(time)
  length_t = time.shape[0]
  index_t = np.arange(length_t)
  
  write_time(dbFile, time, index_t, compression)


def run_script():
  
  args = getArgs()
  dir_path = args.dir_path
  #dir_path = "/raid/home/ksansom/caseFiles/ultrasound/cases/DSI006DA/fluent_4PL"

  search_name = args.search_name
  #search_name = "DSI006DA_4PL_ascii-*"
  
  out_file_name = "{0:s}.hdf5".format(args.output_name)
  #out_file_name = "DSI006DA_4PL.hdf5"

  stop_n = args.n_steps
  #stop_n = 849

  out_path_dir = "post_proc"


  out_path = os.path.join(dir_path, out_path_dir)
  if not os.path.exists(out_path):
    print("creating path directory")
    os.makedirs(out_path)


  sol_files = glob.glob(os.path.join(dir_path, search_name))
  sort_nicely(sol_files)
  path, file_name = os.path.split(sol_files[0])
  split_name = file_name.split('-')
  t_init = float(split_name[-1]) #+ float(restart)*ts
  print("initial time: {0:.4f}".format(t_init))
  
  out_file_path = os.path.join(out_path, out_file_name)
  # Allocate arrays for the fluctuations
  if os.path.isfile(out_file_path):
    print("HDF5 file already exists. It it will be overwritten.")
    os.remove(out_file_path)

  dbFile = h5py.File(out_file_path, 'w')
  #dbFile = h5py.File(out_file_path, 'a', driver='mpio', comm=MPI.COMM_WORLD)

  compression = "lzf"
  #compression = None
  write_binary(dbFile, sol_files, stop_n, compression)


if ( __name__ == '__main__' ):
    run_script()
