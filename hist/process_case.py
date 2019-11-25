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

  # top_dir = "/Volumes/SD/caseFiles"
  top_dir = "/media/store/krs/caseFiles"
  # top_dir = "/media/sansomk/510808DF6345C808/caseFiles"

  df = pd.read_pickle(os.path.join(top_dir, case_file))

  # print(df.head())

  relabel_paths = True
  in_dir = "vwi_proj"
  out_dir = "vwi_proc"
  trans_dir = "vwi_trans"
  image_dir = "images"
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

  # create a new graph
  G = nx.DiGraph()
  n_rows = len(df.index)

  for i in range(n_rows):

    if (i in G):
      f_r = G.nodes[i]['row_data']
      f_pg_info = G.nodes[i]['page_data']
    else:
      # this is the fixed image
      f_r = df.iloc[i]  # the data
      f_pg_info = get_additional_info(f_r)
      G.add_node(i, row_data=f_r, page_data=f_pg_info)
    for j in range(i - epsilon, i + epsilon):
      # this should keep from registring with the same image
      if (j > 0) and (j < n_rows) and (j != i):
        # this is the moving image
        if (j in G):
          t_r = G.nodes[j]['row_data']
          t_pg_info = G.nodes[j]['page_data']
        else:
          # this is the fixed image
          t_r = df.iloc[j]  # the data
          t_pg_info = get_additional_info(t_r)
          G.add_node(j, row_data=t_r, page_data=t_pg_info)



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
        #print(initial_params["best_metric"])

        best_reg_s1 = stage_1_transform(
            reg_dict=reg_n[reg_key], n_max=512, init_angle=initial_params
        )
        # print(
        #     best_reg_s1["transform"].GetParameters(),
        #     best_reg_s1["transform"].GetFixedParameters(),
        # )

        best_reg_s1b, affine_fig = stage_1b_transform(
            reg_dict=reg_n[reg_key], n_max=1024, initial_transform=best_reg_s1
        )
        
        fig_name = os.path.join(image_dir, "fig_affine_{0}_{1}.png".format(i,j))
        fig_path = os.path.join(top_dir, fig_name)
        affine_fig.savefig(fig_path)


        affine_name = os.path.join(trans_dir, "affine_{0}_{1}.h5".format(i,j))
        transform_path = os.path.join(top_dir, affine_name)
        sitk.WriteTransform(best_reg_s1b["transform"], transform_path)

        G.add_edge(i, j, measure = best_reg_s1b["measure"],
                  transform = best_reg_s1b["transform"],
                  tiff_page = best_reg_s1b["tiff_page"],
                  transform_file_name = affine_name )

    
        # best_reg_s2 = stage_2_transform(reg_dict=reg_n[-1], n_max=512, initial_transform=best_reg_s1b)
        # best_reg_s3 = stage_3_transform(reg_dict=reg_n[-1], n_max=2048, initial_transform=best_reg_s2)

        #print(best_reg_s1)
        #print()
        #print(best_reg_s1b)
        #quit()
        #best_reg_s2 = stage_2_transform(reg_dict=reg_n[-1], n_max=512, initial_transform=best_reg_s1b)
        #best_reg_s3 = stage_3_transform(reg_dict=reg_n[-1], n_max=2048, initial_transform=best_reg_s2)
        
        #print(best_reg_s1["measure"], best_reg_s1b["measure"])
  pickle_path = os.path.join(top_dir, case_file.split(".")[0] + ".gpkl" )
  nx.write_gpickle(G, pickle_path)

if __name__ == "__main__":
  main()

