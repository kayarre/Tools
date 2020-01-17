# import pickle
# import pyvips
import os
import pandas as pd

# import numpy as np
import SimpleITK as sitk
import networkx as nx
import pickle
import copy

# import itk
import matplotlib.pyplot as plt

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging

logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)

# from read_vips import parse_vips
from utils import get_additional_info
from utils import _calculate_composite
from utils import read_tiff_image
from utils import get_mean_edges
from utils import resample_rgb
from stage_1_registration import stage_1_transform
from stage_1_parallel import stage_1_parallel_metric
from stage_1b_registration import stage_1b_transform
from stage_1c_registration import stage_1c_transform
from stage_2_registration import stage_2_transform
from stage_3_registration import stage_3_transform


def main():
  # this register the cropped images

  # df_path = "/Volumes/SD/caseFiles/vwi_proj/process_df.pkl"
  # crop_dir = '/Volumes/SD/caseFiles/vwi_proc'
  # csv = pd.read_csv(os.path.join(crop_dir,  "case_1.csv"))
  # df = pd.read_pickle(os.path.join(crop_dir, "case_1.pkl"))

  case_file = "case_1.pkl"

  top_dir = "/Volumes/SD/caseFiles"
  #top_dir = "/media/store/krs/caseFiles"
  # top_dir = "/media/sansomk/510808DF6345C808/caseFiles"

  df = pd.read_pickle(os.path.join(top_dir, case_file))

  # print(df.head())

  #relabel_paths = True
  #in_dir = "vwi_proj"
  #out_dir = "vwi_proc"
  trans_dir = "vwi_trans"
  image_dir = "images"
  test_dir = "test"
  resample_dir = "resample"
  #mask_dir = "masks"
  #print(df.head())
  #print(df.columns)
  # print(df["Image_ID"].values.dtype)

  # study_id = "1"

  # test_reg = register_series()

  # this is the registration loop
  reference_index = 0
  reg_n = {}
  epsilon = 2
  lambda_ = 1.0

  bad_keys = {(23,21) : 1.030835089459151, (10,11) : 4.908738521234052 } #, (11,10) :]

  # create a new graph
  G = nx.DiGraph()
  n_rows = len(df.index)
  init_max = 256
  stage_1_max = 512
  elstix_max = 1024

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
        # if (reg_key not in [(10,11), (23,21)]):
        #   continue

        initial_params, fig_list = stage_1_parallel_metric(
            reg_dict=reg_n[reg_key], n_max=init_max
        )
        #print(initial_params)
        max_pix = copy.deepcopy(init_max)
        for fig_dict in fig_list:
          for key, fig in fig_dict.items():
            fig_name = os.path.join(test_dir, "fig_init_{0}_{1}_{2}_{3}.png".format(i, j, key, int(max_pix)))
            fig_path = os.path.join(top_dir, fig_name)
            fig.savefig(fig_path)
            plt.close(fig)
          max_pix /=2
        # print(initial_params)

        #print(initial_params)
        #print(init_params_flip)
        print(initial_params["best_metric"], initial_params["best_angle"],
              initial_params["best_metric_type"])
        #print(init_params_flip["best_metric"], init_params_flip["best_angle"])
        # quit()
        if (reg_key in bad_keys.keys()):
          initial_params["best_angle"] = bad_keys[reg_key]
        best_reg_s1, rigid_fig = stage_1_transform(
            reg_dict=reg_n[reg_key], n_max=stage_1_max, init_params=initial_params
        )
        # print(
        #     best_reg_s1["transform"]
        # )
        best_reg_s1b, affine_fig = stage_1b_transform(
            reg_dict=reg_n[reg_key], n_max=elstix_max, initial_transform=best_reg_s1
        )
        # print(
        #     best_reg_s1b["transform"]
        # )
        # if (reg_key in bad_keys.keys()):
        #   best_reg_s1b, affine_fig = stage_1b_transform(
        #       reg_dict=reg_n[reg_key], n_max=elstix_max, initial_transform=best_reg_s1
        #   )
        # else:
        #   best_reg_s1b, affine_fig = stage_1c_transform(
        #       reg_dict=reg_n[reg_key], n_max=elstix_max, initial_transform=best_reg_s1
        #   )

        fig_name = os.path.join(image_dir, "fig_rigid_{0}_{1}.png".format(i,j))
        fig_path = os.path.join(top_dir, fig_name)
        rigid_fig.savefig(fig_path)
        plt.close(rigid_fig)

        fig_name = os.path.join(image_dir, "fig_affine_{0}_{1}.png".format(i,j))
        fig_path = os.path.join(top_dir, fig_name)
        affine_fig.savefig(fig_path)
        plt.close(affine_fig)

        affine_name = os.path.join(trans_dir, "affine_{0}_{1}.h5".format(i,j))
        transform_path = os.path.join(top_dir, affine_name)
        sitk.WriteTransform(best_reg_s1["transform"], transform_path)

        abs_ij = abs(i-j)
        # this is the metric from the possum framework
        weight = (1.0 + best_reg_s1b["measure"]) * abs_ij * (1.0 + lambda_)**(abs_ij) 
        G.add_edge(i, j, weight = weight,
                  measure = best_reg_s1["measure"],
                  transform = best_reg_s1["transform"],
                  tiff_page = best_reg_s1["tiff_page"],
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

  # save the registration data
  reg_path = os.path.join(top_dir, case_file.split(".")[0] + "_reg_data.pkl" )
  with open(reg_path, 'wb') as f:
    pickle.dump(reg_n, f)

  # remove transforms in case they can't be pickled
  new_G = G.copy()
  for n1, n2, d in new_G.edges(data=True):
    for att in ["transform"]:
        nothing = d.pop(att, None)
  
  pickle_path2 = os.path.join(top_dir, case_file.split(".")[0] + "_2.gpkl" )
  nx.write_gpickle(new_G, pickle_path2)

  pickle_path = os.path.join(top_dir, case_file.split(".")[0] + ".gpkl" )
  try:
    nx.write_gpickle(G, pickle_path)
  except Exception as e:
    #except pickle.PicklingError as e:
    print(" Cannot pickle this thing {0}".format(e))
  
  # TODO make another script that can generate this from saved transforms and graph 
  for j in range(n_rows):
    # j is the moving image
    trans_list = _calculate_composite(G, reference_index, j)
    # Instanciate composite transform which will handle all the partial
    # transformations.
    composite_transform = sitk.Transform(2, sitk.sitkEuler )
    # Fill the composite transformation with the partial transformations:
    for transform in trans_list:
        composite_transform.AddTransform(transform)

    reg_key = (reference_index, j)
    if (reg_key in reg_n.keys()):
      f_sitk, t_sitk = read_tiff_image(reg_n[reg_key], page_index = 4)
      new_image = resample_rgb(composite_transform,
                               f_sitk,
                               t_sitk,
                               mean = get_mean_edges(t_sitk)
                              )
      resample_image =  os.path.join(top_dir, resample_dir, "resample_affine_{0}.png".format(j))
      writer = sitk.ImageFileWriter()
      writer.SetFileName(resample_image)
      writer.Execute(new_image)

if __name__ == "__main__":
  main()

