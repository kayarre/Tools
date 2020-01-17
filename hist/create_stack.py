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

import utils


def main():
  # this register the cropped images

  # df_path = "/Volumes/SD/caseFiles/vwi_proj/process_df.pkl"
  # crop_dir = '/Volumes/SD/caseFiles/vwi_proc'
  # csv = pd.read_csv(os.path.join(crop_dir,  "case_1.csv"))
  # df = pd.read_pickle(os.path.join(crop_dir, "case_1.pkl"))

  case_file = "case_1.pkl"
  reference_index = 0

  top_dir = "/Volumes/SD/caseFiles"
  #top_dir = "/media/store/krs/caseFiles"
  # top_dir = "/media/sansomk/510808DF6345C808/caseFiles"

  df = pd.read_pickle(os.path.join(top_dir, case_file))
  n_rows = len(df.index)
  # print(df.head())

  #relabel_paths = True
  #in_dir = "vwi_proj"
  #out_dir = "vwi_proc"
  trans_dir = "vwi_trans"
  #image_dir = "images"
  #test_dir = "test"
  resample_dir = "resample"
  #mask_dir = "masks"
  #print(df.head())
  #print(df.columns)
  # print(df["Image_ID"].values.dtype)

  # save the registration data
  reg_path = os.path.join(top_dir, case_file.split(".")[0] + "_reg_data.pkl" )  
  reg_n = pickle.load(open(reg_path, "rb"))


  graph_name = case_file.split(".")[0] + "_2.gpkl"
  graph_path = os.path.join(top_dir, graph_name)
  G_read = nx.read_gpickle(graph_path)

  G = G_read.copy()
  for n1, n2, d in G.edges(data=True):
    # print(d["transform_file_name"])
    transform_path = os.path.join(top_dir, d["transform_file_name"])
    trans_ = sitk.ReadTransform(transform_path)
    G.edges[(n1,n2)]["transform"] = trans_
    # print("yo", trans_.GetDimension(), trans_.GetFixedParameters(),
    #         trans_.GetParameters(), trans_.GetName())

  writer = sitk.ImageFileWriter()
  # TODO make another script that can generate this from saved transforms and graph
  ref_key = (reference_index, reference_index)
  reg_dict = {}
  for j in range(n_rows):
    print(j)
    data = df.iloc[j]
    # j is the moving image
    trans_list = utils._calculate_composite(G, reference_index, j)
    #print(trans_list)
    # Instantiate composite transform which will handle all the partial
    # transformations.
    composite_transform = sitk.Transform(2, sitk.sitkEuler)
    #composite_transform = sitk.Transform(2, sitk.sitkComposite)
    # Fill the composite transformation with the partial transformations:
    for transform in trans_list:
      #print(transform)
      # print("test", transform.GetDimension(), transform.GetFixedParameters(),
      #       transform.GetParameters(), transform.GetName())
      transform.FlattenTransform()
      composite_transform.AddTransform(transform)
    #print(dir(composite_transform))
    #print(help(composite_transform.FlattenTransform))
    #print(help(composite_transform))

    #quit()
    composite_transform.FlattenTransform()
    # if ( j > 2):
    #   print(composite_transform)
    #   quit()
    
    # print("yo", composite_transform.GetDimension(), composite_transform.GetFixedParameters(),
    #         composite_transform.GetParameters(), composite_transform.GetName())

    reg_key = (reference_index, j)
    test_chain = utils._get_transformation_chain(G, reference_index, j)
    print(reg_key)
    print(test_chain[0])
    if (reg_key in [ref_key]):
      # need hack to get a registration with the reference image in it
      # if this doesn't work then you have big problems
      for key in reg_n.keys():
        if (reg_key[0] == key[0] ):
          base_key = key
          break

      f_sitk = utils.read_1_tiff_image(reg_n[base_key], page_index = 5)
      #print(f_sitk.GetSize())
      new_image = utils.resample_1_rgb(composite_transform.GetInverse(),
                                       f_sitk,
                                       mean = utils.get_mean_edges(f_sitk) )
      resample_im = "resample_affine_{0}.png".format(j)
      resample_path = os.path.join(top_dir, resample_dir, resample_im)
      print(resample_path)
      writer.SetFileName(resample_path)
      writer.Execute(new_image)
      reg_dict[reg_key] = dict(png_name = resample_im)

    else:
      #key_test = tuple(reversed(test_chain[0]))
      key_test = tuple(test_chain[0])
      print(key_test)
      t_sitk = utils.read_1_tiff_image(reg_n[key_test], page_index = 5)
      #print(t_sitk.GetSize())
      print(t_sitk)
      print(composite_transform)
      quit()
      new_image = utils.resample_1_rgb(composite_transform.GetInverse(),
                               t_sitk,
                               mean = utils.get_mean_edges(t_sitk)
                              )

      #checkerboard = sitk.CheckerBoardImageFilter()
      #check_im = checkerboard.Execute(f_sitk, new_image, (8,8))

      # utils.display_images(
      #     fixed_npa=sitk.GetArrayViewFromImage(f_sitk),
      #     moving_npa=sitk.GetArrayViewFromImage(new_image),
      #     checkerboard=sitk.GetArrayViewFromImage(check_im),
      #     show=True
      # )
      #print(new_image)

      #test_im = sitk.Image([10,10], sitk.sitkVectorUInt8, 3)
      #print(test_im)
      #quit()
      resample_im = "resample_affine_{0}.png".format(j)
      resample_path = os.path.join(top_dir, resample_dir, resample_im)
      print(resample_path)
      writer.SetFileName(resample_path)
      writer.Execute(new_image)
      reg_dict[reg_key] = dict(png_name = resample_im)

      resample_orig = "resample_orig_{0}.png".format(j)
      resample_path = os.path.join(top_dir, resample_dir, resample_orig)
      #print(resample_path)
      writer.SetFileName(resample_path)
      writer.Execute(t_sitk)


if __name__ == "__main__":
  main()
