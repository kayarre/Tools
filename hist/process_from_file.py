# import pickle
# import pyvips
import os
import pandas as pd

# import numpy as np
import SimpleITK as sitk
import networkx as nx
import pickle

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

  trans_dir = "vwi_trans"
  image_dir = "images"
  resample_dir = "resample"

  n_rows = len(df.index)


  # this is the registration loop
  reference_index = 0
  pickle_path2 = os.path.join(top_dir, case_file.split(".")[0] + "_2.gpkl" )

  G = nx.read_gpickle(pickle_path2)
  
  # TODO make another script that can generate this from saved transforms and graph 
  for j in range(n_rows):
    # j is the moving image
    trans_list = _calculate_composite(G, reference_index, j)
    # Instanciate composite transform which will handle all the partial
    # transformations.
    composite_transform = sitk.Transform()
    # Fill the composite transformation with the partial transformations:
    for transform in trans_list:
        composite_transform.AddTransform(transform)

    reg_key = (reference_index, j)
    if (reg_key in reg_n.keys()):
      f_sitk, t_sitk = read_tiff_image(reg_n[reg_key], page_index=4)
      new_image = resample_rgb(composite_transform,
                               f_sitk,
                               t_sitk,
                               mean=get_mean_edges(t_sitk)
                              )
      resample_image =  os.path.join(top_dir, resample_dir, "resample_affine_{0}.png".format(j))
      writer = sitk.ImageFileWriter()
      writer.SetFileName(resample_image)
      writer.Execute(new_image)

if __name__ == "__main__":
  main()

