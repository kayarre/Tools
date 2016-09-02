# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import sys
import re
import glob
import math
import copy as cp
import multiprocessing as mp
import tarfile
import pickle
import gzip


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

def load_pickle(pikl_file_path):
  try:
    with open(pikl_file_path, "rb") as pkl_f:
      processed = pickle.load(pkl_f)
      completed = pickle.load(pkl_f)
  except Exception as e: 
    print(str(e))
    print("pickle file empty creating empty variables")
    processed = {}
    completed = []
  else:
    print('Read the pickle file from the current directory')
  
  return processed, completed

def calc_one_step(out_file_name, file_list, out_path, current_step,
    restart, cycles, ts, remove_files, pikl_file_path):

  f_processed, f_completed = load_pickle(pikl_file_path)
  print('tar the files')
  tar_file_name = "{0}_sol_{1:.4f}.tar".format(out_file_name, float(current_step)*ts)
  print(tar_file_name)
  if(tar_file_name in f_completed):
    # this tar file is done
    print("this one is done")
  else:
    with tarfile.open(name=os.path.join(out_path, tar_file_name), mode="a:") as tar:
      for file_path in file_list:
        if(tar_file_name in f_processed.keys()):
          if(file_path in f_processed[tar_file_name][1]):
            #already processed this file
            continue
        print(os.path.split(file_path)[-1])
        if( os.path.isfile(file_path)):
          tar.add(file_path, os.path.split(file_path)[-1])
          if(tar_file_name in f_processed.keys()):
            f_processed[tar_file_name][1].append(file_path)
            f_processed[tar_file_name][0][0] += 1
          else:
            f_processed[tar_file_name] = []
            f_processed[tar_file_name].append([1])
            f_processed[tar_file_name].append([file_path])
        else:
          print('file doesnt exist yet')
          
    
    for file_key in f_processed.keys():
      if(f_processed[file_key][0] == cycles):
        gz_file_name = "{0}.gz".format(file_key)
        with gzip.open(os.path.join(out_path, gz_file_name), "wb") as gz:
          gz.write(os.path.join(out_path, file_key))
        f_completed.append(file_key)

    with open(pikl_file_path, "wb") as pkl_f:
        pickle.dump(f_processed, pkl_f, -1)
        pickle.dump(f_completed, pkl_f, -1)


  print("step {0} complete".format(current_step))
  
      

def remove_files(file_list, remove_files=False):
  if (remove_files == True):
    print('removing files')
    for f_idx, file_path in enumerate(file_list):
      #print(os.path.split(file_path)[-1])
      try:
        os.remove(file_path)
      except Exception, err:
        print('Exception:{0}'.format(err)) 
        continue
      #for p, vx, vy, vz, s_p, s_vx, s_vy, s_vz in zip(
      #  p_avg, vx_avg, vy_avg, vz_avg, sig_p, sig_vx, sig_vy, sig_vz):

def run_script():
  remove_files = False
  dir_path = "/raid/home/ksansom/caseFiles/ultrasound/fistula_repeat/fluentHyak/fromHyak/temp"
  out_path = "/raid/home/ksansom/caseFiles/ultrasound/fistula_repeat/fluentHyak/fromHyak/tartest"
  search_name = "fistula_fluent-*"
  out_file_name = "fistula"
  remove_files = False
  pikl_file_path = os.path.join(out_path, "tarfiles.pkl")
  
  if not os.path.exists(out_path):
    print("creating path directory")
    os.makedirs(out_path)
  wave_len = 0.85 # s
  ts = 0.001 #s
  cycles = 10
  restart = 0
  stop_n = 850 # time steps, max is wave_len/ts
  t_orig = 1.700
  nprocs = 1
  dry_run = False
  steps = int(wave_len/ts) # int
  print('{0} steps in each cycle'.format(steps))
  print('restart at step {0}'.format(restart))
  sol_files = glob.glob(os.path.join(dir_path, search_name))
  sort_nicely(sol_files)
  path, file_name = os.path.split(sol_files[0])
  split_name = file_name.split('-')
  t_init = float(split_name[-1]) #+ float(restart)*ts
  print("initial time: {0:.4f}".format(t_init))
  
  pool = mp.Pool(processes=nprocs)
  for i in range(steps):
    if ( i < restart):
      continue
    if (i > stop_n):
      continue

    time_list = []
    file_names = []
    for j in range(cycles):
      time_list.append("{0:.4f}".format(
        float(i)*ts + t_orig + float(j)*wave_len))
    print(time_list)
    for t in time_list:
      file_names.append('-'.join(split_name[0:-1]) + '-' + t)
    print(file_names)
    
    file_list = [os.path.join(path, p) for p in file_names]
    print(out_file_name)
    if (dry_run == False):
      #pool.apply_async(calc_one_step,
      #  args=(out_file_name, file_list, out_path, i, restart, cycles, ts, remove_files, pikl_file_path))
      calc_one_step(out_file_name, file_list, out_path, i, restart, cycles, ts, remove_files, pikl_file_path)
      
    else:
      print('dry run')

  
  # run processes
  #pool.close()
  #pool.join()
  
  # Get process results from the output queue
  #results = [output.get() for p in processes]
  #print(results)
    
if ( __name__ == '__main__' ):
    run_script()
