#mpi_ascii.py
from mpi4py import MPI
import numpy as np
import pandas as pd
import os
import h5py

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

      data = pd.read_csv(file_, sep='\s+', usecols=np.arange(1,7)).values
      if(file_idx == 0):
        coords = data[:,0:3]
        v = data[:,3:6]
        initialize_write(dbFile, file_idx, split_name[-1], coords, v, compression)
      else:
        v = data[:,3:6]
        write_fields(dbFile, file_idx, split_name[-1], v, compression)

  time = np.asarray(time)
  length_t = time.shape[0]
  index_t = np.arange(length_t)

  write_time(dbFile, time, index_t, compression)


def run_script():
  rank = MPI.COMM_WORLD.Get_rank()
  size = MPI.COMM_WORLD.Get_size()
  name = MPI.Get_processor_name()

  # ******************************
  # actual (serial) work goes here
  # ******************************

  #python ascii_2_binary.py --dir_path=/raid/home/ksansom/caseFiles/ultrasound/cases/DSI006DA/fluent_4PL --search_name=DSI006DA_4PL_ascii-* --output_name=DSI006DA_4PL --n_steps=2


  dir_path = "/raid/home/ksansom/caseFiles/ultrasound/cases"
  sub_dir = "fluent"
  file_list_name = "file_list"
  out_path_dir = "post_proc"

  case_dict = {"DSI002CARc" : [1, 635],
                    "DSI003LERd"  : [2, 836],
                    "DSI006DA"    : [3, 849],
                    "DSI007LERb"  : [4, 644],
                    "DSI009CALb"  : [5, 691],
                    "DSI010CALb"  : [6, 1030],
                    "DSI010LERd"  : [7, 908],
                    "DSI011CARe"  : [8, 1769],
                    "DSI015DALd"  : [9, 1930]
                    }
  #print("Hello, world! This is rank {0:d} of {1:d} running on {2:s}".format(rank, size, name))
  
  case_id = case_dict.keys()
  length_cases = len(case_dict.keys())

  if(rank == 0):
    print(length_cases)
  #length_cases = 2
  if(length_cases > size):
    print("Not enough processors to complete")
    MPI.COMM_WORLD.Abort()
  else:
    for idx, case in enumerate(case_id):
      #here is one case execute for that rank
      #print("yo", idx, case)
      if(rank > length_cases):
        continue
      elif(rank+1 == idx):
        print("run case {0:s} on rank {1:d}".format(case, rank))
        ascii_path = os.path.join(dir_path, case, sub_dir)
        file_list_path = os.path.join(ascii_path, file_list_name)

        out_file_name = "{0:s}.hdf5".format(case)
        out_path = os.path.join(ascii_path, out_path_dir)
        #out_file_path = os.path.join(out_path, out_file_name)

        if not os.path.exists(out_path):
          print("creating path directory")
          os.makedirs(out_path)


        sol_names = pd.read_csv(file_list_path, header=None).values
        sol_files = [os.path.join(ascii_path, str(i[0])) for i in sol_names]

        #make sure all the files exist
        for fname in sol_files:
          if(not os.path.isfile(fname)):
            print("There is a missing file in case {0:s}".format(case))
            print(fname)
            continue

        out_file_path = os.path.join(out_path, out_file_name)
        # Allocate arrays for the fluctuations
        if os.path.isfile(out_file_path):
          print("HDF5 file already exists. It it will be overwritten.")
          try:
            os.remove(out_file_path)
          except FileNotFoundError:
            print("Error trying to delete: {0:s}".format(out_file_path))
        dbFile = h5py.File(out_file_path, 'w')
        #dbFile = h5py.File(out_file_path, 'a', driver='mpio', comm=MPI.COMM_WORLD)

        compression = "lzf"
        #compression = None
        #write_binary(dbFile, sol_files, case_dict[case][1], compression)
        write_binary(dbFile, sol_files, 2, compression)

  MPI.COMM_WORLD.Barrier()   # wait for everybody to synchronize _here_      

if ( __name__ == '__main__' ):
  run_script()


