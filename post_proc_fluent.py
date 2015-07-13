# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import glob
import math
from mpl_toolkits.mplot3d import Axes3D
import copy as cp
import multiprocessing as mp


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

def calc_one_step(file_list, out_path, current_step, restart, cycles, ts):
  x_coords = []
  y_coords = []
  z_coords = []  
  nodes = []
  vx_avg = []
  vy_avg = []
  vz_avg = []
  p_avg = []
  sig_p = []
  sig_vx = []
  sig_vy = []
  sig_vz = []
  data_f_list = []
  print('calculate average')
  for f_idx, file_path in enumerate(file_list):
    print(os.path.split(file_path)[-1])
    if os.path.isfile(file_path):
      
      data_f_list.append(open(file_path, 'rU'))
      for idx, line in enumerate(data_f_list[-1]):
        #print(line)
        #line = re.sub(',', ' ', line)
        if (idx == 0):
          f_vars = line.split()
        else:
          nums = line.split()
          #print(nums)
          if (f_idx == 0):
            nodes.append(int(nums[0]))
            if (current_step == restart):
              x_coords.append(float(nums[1]))
              y_coords.append(float(nums[2]))
              z_coords.append(float(nums[3]))

            else:
              if (int(nums[0]) != nodes[idx-1]):
                print("nodes don't match {0} {1}".format(int(nums[0]),
                      nodes[idx-1]))

            p_avg.append(float(nums[4]))
            vx_avg.append(float(nums[5]))
            vy_avg.append(float(nums[6]))
            vz_avg.append(float(nums[7]))
            #print(p_avg[-1], vx_avg[-1], vy_avg[-1], vz_avg[-1])
            #print(p_avg[idx-1], vx_avg[idx-1], vy_avg[idx-1], vz_avg[idx-1])
          else:
            if (int(nums[0]) != nodes[idx-1]):
              print("nodes don't match {0} {1}".format(int(nums[0]),
                    nodes[idx-1]))
            p_avg[idx-1] += float(nums[4])
            vx_avg[idx-1] += float(nums[5])
            vy_avg[idx-1] += float(nums[6])
            vz_avg[idx-1] += float(nums[7])
            #print(p_avg[idx-1], vx_avg[idx-1], vy_avg[idx-1], vz_avg[idx-1])
      data_f_list[-1].seek(0)
  p_avg = [p / float(cycles) for p in p_avg]
  vx_avg = [p / float(cycles) for p in vx_avg]
  vy_avg = [p / float(cycles) for p in vy_avg]
  vz_avg = [p / float(cycles) for p in vz_avg]
  #plt.scatter(range(nodes[-1]), vx_avg)
  #plt.scatter(range(nodes[-1]), vy_avg)
  #plt.scatter(range(nodes[-1]), vz_avg)
  #plt.scatter(range(nodes[-1]), p_avg)
  #return             
  # standard deviation
  print('calculate standard deviation')
  for f_idx, data_f in enumerate(data_f_list):
    #for f_idx, file_n in enumerate(file_list):
    print(os.path.split(file_list[f_idx])[-1])
    #file_path = os.path.join(path, file_n)
    #if os.path.isfile(file_path):
    #with open(file_path, 'rU') as data_f:
    for idx, line in enumerate(data_f):
      #print(line)
      #line = re.sub(',', ' ', line)
      if (idx == 0):
        f_vars = line.split()
        #print(f_vars)
      else:
        nums = line.split()
        n = int(nums[0])
        if n != nodes[idx-1]:
          print("nodes don't match {0} {1}".format(n, nodes[idx-1]))
       
        if (f_idx == 0):              
          sig_p.append(pow((float(nums[4]) - p_avg[idx-1]), 2.0))
          sig_vx.append(pow((float(nums[5]) - vx_avg[idx-1]), 2.0))
          sig_vy.append(pow((float(nums[6]) - vy_avg[idx-1]), 2.0))
          sig_vz.append(pow((float(nums[7]) - vz_avg[idx-1]), 2.0))
        else:
          sig_p[idx-1] += pow((float(nums[4]) - p_avg[idx-1]), 2.0)
          sig_vx[idx-1] += pow((float(nums[5]) - vx_avg[idx-1]), 2.0)
          sig_vy[idx-1] += pow((float(nums[6]) - vy_avg[idx-1]), 2.0)
          sig_vz[idx-1] += pow((float(nums[7]) - vz_avg[idx-1]), 2.0)
    data_f.close()
  n_bias = 1.0
  sig_p = [np.sqrt(p / (float(cycles) - n_bias)) for p in sig_p]
  sig_vx = [np.sqrt(p / (float(cycles) - n_bias)) for p in sig_vx]
  sig_vy = [np.sqrt(p / (float(cycles) - n_bias)) for p in sig_vy]
  sig_vz = [np.sqrt(p / (float(cycles) - n_bias)) for p in sig_vz] 
  #plt.plot(range(nodes[-1]), sig_p)
  #plt.plot(range(nodes[-1]), sig_vx)
  #plt.plot(range(nodes[-1]), sig_vy)
  #plt.plot(range(nodes[-1]), sig_vz)
  #return

  if (current_step == 0):
    print("write coordinate file")
    with open(os.path.join(out_path, 'coordinates.txt'), 'w') as coord_f:
      coord_f.write(" ".join(f_vars[0:4]))
      coord_f.write("\n")
      for n, x, y, z in zip(nodes, x_coords, y_coords, z_coords):
        coord_f.write(
          "{0:10n} {1:16.9E} {2:16.9E} {3:16.9E}\n".format(
          n, x, y, z))

  time_file_name = "carotid_sol_{0:.4f}.txt".format(float(current_step)*ts)
  with open(os.path.join(out_path, time_file_name), 'w') as params_f:
    print("write parameter file")
    params_f.write(f_vars[0] + " ")
    params_f.write("average-absolute-pressure average-x-velocity ")
    params_f.write("average-y-velocity average-z-velocity ")
    params_f.write("sigma-absolute-pressure sigma-x-velocity ")
    params_f.write("sigma-y-velocity sigma-z-velocity")
    params_f.write("\n")
    for n, p, x, y, z, sp, sx, sy, sz in zip(nodes, p_avg, vx_avg, vy_avg,
                                             vz_avg, sig_p, sig_vx,
                                             sig_vy, sig_vz): 
      params_f.write("{0:10n} ".format(n))
      params_f.write("{0:16.9E} {1:16.9E} {2:16.9E} {3:16.9E} ".format(
                    p, x, y, z))
      params_f.write("{0:16.9E} {1:16.9E} {2:16.9E} {3:16.9E}\n".format(
                    sp, sx, sy, sz)) 
                    
  print("step {0} complete".format(current_step))
  #output.put((current_step, "step {0} complete".format(current_step)))
  
      

def remove_files(file_list, remove_files=False):
  if (remove_files == True):
    print('removing files')
    for f_idx, file_n in enumerate(file_list):
      #print(file_n)
      file_path = os.path.join(path, file_n)
      try:
        os.remove(file_path)
      except Exception, err:
        print('Exception:{0}'.format(err)) 
        continue
      #for p, vx, vy, vz, s_p, s_vx, s_vy, s_vz in zip(
      #  p_avg, vx_avg, vy_avg, vz_avg, sig_p, sig_vx, sig_vy, sig_vz):

def run_script():
  remove_files = False
  dir_path = "/raid/home/ksansom/caseFiles/ultrasound/fistula/fluent"
  out_path = "/raid/home/ksansom/caseFiles/ultrasound/fistula/fluent/fluent_post_proc"
  search_name = "fistula_fluent_ascii-*"
  
  
  
  if not os.path.exists(out_path):
    print("creating path directory")
    os.makedirs(out_path)
  #out_path = "/home/vnc/carotid/conv_out/"
  wave_len = 0.85 # s
  ts = 0.001 #s
  cycles = 10
  restart = 0
  stop_n = 850 # time steps, max is wave_len/ts
  t_orig = 1.700
  nprocs = 24
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
  
  processes = mp.Pool()
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
    
    if (dry_run == False):
      processes.apply_async(calc_one_step,
        args=(file_list, out_path, i, restart, cycles, ts))
      #calc_one_step(output, file_list, out_path, i, restart, cycles, ts)
      
    else:
      print('dry run')
    #print(p_avg[10], vx_avg[10], vy_avg[10], vz_avg[10])
    #print(sig_p[10], sig_vx[10], sig_vy[10], sig_vz[10])
  
  # run processes
  processes.close()
  processes.join()
  
  # Get process results from the output queue
  results = [output.get() for p in processes]
  print(results)
    
if ( __name__ == '__main__' ):
    run_script()
