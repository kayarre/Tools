# import pickle
# import pyvips
import os
import pandas as pd

# import numpy as np
import SimpleITK as sitk
import networkx as nx

# import itk
# import matplotlib.pyplot as plt

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging

logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)

# from read_vips import parse_vips
from utils import get_additional_info
from stage_1_registration import stage_1_transform
from stage_1_parallel import stage_1_parallel_metric
from stage_1b_registration import stage_1b_transform
from stage_2_registration import stage_2_transform
from stage_3_registration import stage_3_transform


def main():
  # this register the cropped images

  # df_path = "/Volumes/SD/caseFiles/vwi_proj/process_df.pkl"
  # crop_dir = '/Volumes/SD/caseFiles/vwi_proc'
  # csv = pd.read_csv(os.path.join(crop_dir,  "case_1.csv"))
  # df = pd.read_pickle(os.path.join(crop_dir, "case_1.pkl"))

  case_file = "case_1.pkl"
  # top_dir = "/media/store/krs/caseFiles"
  # top_dir = "/Volumes/SD/caseFiles"
  top_dir = "/media/sansomk/510808DF6345C808/caseFiles"

  df = pd.read_pickle(os.path.join(top_dir, case_file))

  # print(df.head())

  relabel_paths = True
  in_dir = "vwi_proj"
  out_dir = "vwi_proc"
  #print(df.head())
  #print(df.columns)
  # print(df["Image_ID"].values.dtype)

  # study_id = "1"

  # test_reg = register_series()

  # this is the registration loop
  tmp_index = 0
  reg_n = {}
  epsilon = 3
  lamda = 1.0

  n_rows = len(df.index)

  for i in range(n_rows):
    for j in range(i - epsilon, i + epsilon):
      # this should keep from registring with the same image
      if (j > 0) and (j < n_rows) and (j != i):
        # this is the fixed image
        f_r = df.iloc[i]  # the data
        #print(f_r)
        f_pg_info = get_additional_info(f_r)

        # this is the moving image
        t_r = df.iloc[j]  # the data
        t_pg_info = get_additional_info(t_r)

        # this is the
        reg_key = (i, j)
        reg_n[reg_key] = dict(
            f_row=f_r, t_row=t_r, f_page=f_pg_info, t_page=t_pg_info
        )
        print(reg_key)

        initial_params = stage_1_parallel_metric(
            reg_dict=reg_n[reg_key], n_max=512
        )
        #print(initial_params)
        print(initial_params["best_metric"])

        best_reg_s1 = stage_1_transform(
            reg_dict=reg_n[reg_key], n_max=1024, init_angle=initial_params
        )
        # print(
        #     best_reg_s1["transform"].GetParameters(),
        #     best_reg_s1["transform"].GetFixedParameters(),
        # )

        best_reg_s1b = stage_1b_transform(
            reg_dict=reg_n[reg_key], n_max=2048, initial_transform=best_reg_s1
        )
        
        # best_reg_s2 = stage_2_transform(reg_dict=reg_n[-1], n_max=512, initial_transform=best_reg_s1b)
        # best_reg_s3 = stage_3_transform(reg_dict=reg_n[-1], n_max=2048, initial_transform=best_reg_s2)

        print(best_reg_s1)
        print()
        print(best_reg_s1b)
        quit()
        #best_reg_s2 = stage_2_transform(reg_dict=reg_n[-1], n_max=512, initial_transform=best_reg_s1b)
        #best_reg_s3 = stage_3_transform(reg_dict=reg_n[-1], n_max=2048, initial_transform=best_reg_s2)
        
        print(best_reg_s1["measure"], best_reg_s1b["measure"])

  # for row in df.iterrows():
  #   # pg_info = get_additional_info(row)
  #   # test_reg.update_image(row[1])
  #   # test_reg.Execute()
  #   # help(test_reg.trans_image_filter)

  #   if tmp_index == row[0]:
  #       f_r = row  # the data
  #       f_pg_info = get_additional_info(f_r[1])
  #       continue
  #   else:
  #       t_r = row
  #       t_pg_info = get_additional_info(t_r[1])

  #   # leave the index for now
  #   reg_n.append(
  #       dict(f_row=f_r[1], t_row=t_r[1], f_page=f_pg_info, t_page=t_pg_info)
  #   )
  #   # print(reg_n)
  #   # quit()
  #   # print(template[0], fixed[0])

  #   # stage_1_params = sitk.GetDefaultParameterMap("translation")
  #   # stage_1_params["NumberOfResolutions"] = ['1']
  #   # print(sitk.PrintParameterMap(stage_1_params))
  #   # quit()
  #   initial_params = stage_1_parallel_metric(reg_dict=reg_n[-1], n_max=512)
  #   print(initial_params)
  #   print(initial_params["best_metric"])

  #   best_reg_s1 = stage_1_transform(
  #       reg_dict=reg_n[-1], n_max=1024, init_angle=initial_params
  #   )
  #   print(
  #       best_reg_s1["transform"].GetParameters(),
  #       best_reg_s1["transform"].GetFixedParameters(),
  #   )

  #   best_reg_s1b = stage_1b_transform(
  #       reg_dict=reg_n[-1], n_max=2048, initial_transform=best_reg_s1
  #   )
  #   quit()
  #   # best_reg_s2 = stage_2_transform(reg_dict=reg_n[-1], n_max=512, initial_transform=best_reg_s1b)
  #   # best_reg_s3 = stage_3_transform(reg_dict=reg_n[-1], n_max=2048, initial_transform=best_reg_s2)

  #   print(best_reg_s1["measure"], best_reg_s1b["measure"])

  #   # print(best_reg_s1["measure"], best_reg_s2["measure"], best_reg_s3["measure"])

  #   quit()

  #   f_r = t_r
  #   f_pg_info = t_pg_info
  #   # print(template_index, fixed_index)
  #   tmp_index += 1


if __name__ == "__main__":
  main()

