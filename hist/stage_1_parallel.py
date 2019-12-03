import pickle 
#import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
from multiprocessing.pool import ThreadPool
from functools import partial
import copy

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

import utils #import get_sitk_image, display_images

# This function evaluates the metric value in a thread safe manner
def evaluate_metric(current_rotation, tx, f_image, m_image,
                        metric="mmi", n_bins = 50,
                        f_mask=None, m_mask=None):
  reg_method = sitk.ImageRegistrationMethod()
  if(metric == "mmi"):
    reg_method.SetMetricAsMattesMutualInformation(     
          numberOfHistogramBins=n_bins)
  elif( metric == "ants"):
    reg_method.SetMetricAsANTSNeighborhoodCorrelation(radius=4)
  elif(metric == "mse"):
    reg_method.SetMetricAsMeanSquares()

  if ((f_mask is not None) and (m_mask is not None)):
    reg_method.SetMetricFixedMask(f_mask)
    reg_method.SetMetricMovingMask(m_mask)

  reg_method.SetMetricSamplingStrategy(reg_method.RANDOM)
  reg_method.SetMetricSamplingPercentage(0.2)
  reg_method.SetInterpolator(sitk.sitkLinear)
  current_transform = sitk.Euler2DTransform(tx)
  current_transform.SetAngle(current_rotation)
  #print(current_rotation)
  reg_method.SetInitialTransform(current_transform)

  #reg_method.SetMetricFixedMask(mask)
  #reg_method.SetMetricMovingMask(mask)

  exception = True
  count = 0
  while (exception == True) and (count < 3):
    try:
      res = reg_method.MetricEvaluate(f_image, m_image)
    except RuntimeError as e:
      count += 1
      print("Got an exception\n" + str(e))
      continue
    exception = False

  #print(res)
  if (exception == True):
    return 0.0
  else:
    return res


# Threads of threads ?????
# don't know if this makes sense
def evaluate_metric_extra(current_page,
                          f_image, f_mask,
                          m_image, m_mask, angles):
  page_idx = current_page["index"]
  spacing = ( current_page["mmp_x"], current_page["mmp_y"] )
  # transform numpy array to simpleITK image
  # have set the parameters manually
  im_f = tiff.imread(f_image, key=page_idx)
  f_sitk = utils.get_sitk_image(im_f, spacing)

  im_t = tiff.imread(m_image, key=page_idx)
  t_sitk = utils.get_sitk_image(im_t, spacing)

  im_mask_f = sitk.ReadImage(f_mask)
  im_mask_t = sitk.ReadImage(m_mask)

  # this is the union operator take the maximum of the two images
  #max_mask = sitk.Maximum(im_mask_f, im_mask_t)
  identity = sitk.Transform(im_mask_f.GetDimension(), sitk.sitkIdentity)
  f_mask_resampled = sitk.Resample(im_mask_f, f_sitk,
                                    identity, sitk.sitkNearestNeighbor,
                                    0.0, im_mask_f.GetPixelID())
  t_mask_resampled = sitk.Resample(im_mask_t, t_sitk,
                                    identity, sitk.sitkNearestNeighbor,
                                    0.0, im_mask_t.GetPixelID())
  #cnt_pixels = np.count_nonzero(sitk.GetArrayViewFromImage(mask_resampled))
  n_bins = int(np.cbrt(np.prod(sitk.GetArrayViewFromImage(t_mask_resampled).shape)))

  f_input = sitk.Cast(f_sitk*f_mask_resampled, sitk.sitkFloat32)
  t_input = sitk.Cast(t_sitk*t_mask_resampled, sitk.sitkFloat32)
  # im_mask_t = sitk.ReadImage(m_mask)
  # #print(im_mask_t)

  # t_mask_resampled = sitk.Resample(im_mask_t, t_sitk,
  #                                   identity, sitk.sitkNearestNeighbor,
  #                                   0.0, im_mask_t.GetPixelID())
  initial_transform = sitk.CenteredTransformInitializer(f_input, 
                                                  t_input, 
                                                  sitk.Euler2DTransform(), 
                                                  sitk.CenteredTransformInitializerFilter.MOMENTS)

  res_dict = {}
  
  all_values = {}
  with ThreadPool(len(angles)) as pool:
    for m_type in ["mmi", "ants", "mse"]:
      all_values[m_type] = pool.map(partial(evaluate_metric, 
                                            tx = initial_transform, 
                                            f_image = f_input,
                                            m_image = t_input,
                                            metric = m_type,
                                            n_bins = n_bins),
                                    angles)
  for m_type in ["mmi", "ants", "mse"]:
    res_dict[m_type] = dict(angle = angles[np.argmin(all_values[m_type])],
                            metric = np.min(all_values[m_type])
                            )
  # this leaks if you don't close it
  #p.close()
  # p = ThreadPool(len(angles))
  # p.close()

  #print(all_metric_values)
  #print('best orientation is: ' + str(best_orientation))
  best_values = min(res_dict.values(), key=lambda x:x['metric'])
  best_metric = min(res_dict.keys(), key=lambda x:res_dict[x]['metric'])
  param_test = {}
  param_test[page_idx] = dict(angle = best_values["angle"],
                              metric = best_values["metric"],
                              metric_type = best_metric,
                              size_x = current_page["size_x"],
                              size_y = current_page["size_y"],
                              all = res_dict)

  return param_test

# n_max is the maxmimum picture size for registration
def stage_1_parallel_metric(reg_dict, n_max, count=0):
  #print(fixed[1]["crop_paths"])
  ff_path = reg_dict["f_row"]["crop_paths"]
  ff_mask_path = reg_dict["f_row"]["mask_path"]
  tf_path = reg_dict["t_row"]["crop_paths"]
  tf_mask_path = reg_dict["t_row"]["mask_path"]

  #base_res_x = reg_dict["f_row"]["mpp-x"]
  #base_res_y = reg_dict["f_row"]["mpp-y"]
  #param_test = {}
  # get the best one over 30 rotationss
  n_angles = list(np.linspace(0.0, 2.0 * np.pi * 127.0 / 128.0, 128))
  page_list = []
  for page in reg_dict["f_page"]:
    if( page["size_x"] > n_max):
      continue
    page_list.append(page)

  # print("tst")
  #dr = 2.0*np.pi / (64.0 * len(page_list))
  with ThreadPool(len(page_list)) as p_page:
    list_dicts = p_page.map(partial(evaluate_metric_extra, 
                                    f_image = ff_path,
                                    f_mask = ff_mask_path,
                                    m_image = tf_path,
                                    m_mask = tf_mask_path,
                                    angles = n_angles),
                            page_list)
  # this leaks if you don't close it
  #p_page.close()
  # print("tst")
  result = {}
  metric = 99999999.0
  keep = None
  idx = None
  for d in list_dicts:
    result.update(d)
    for k, data in d.items():
      if (data["metric"] < metric):
        metric = copy.deepcopy(data["metric"])
        keep = copy.deepcopy(d[k])
        idx = copy.deepcopy(k)
  result["n_angles"] = len(n_angles)
  result["best_angle"] = keep["angle"]
  result["best_metric"] = metric
  result["best_page_idx"] = idx
  result["best_metric_type"] = keep["metric_type"]

  return result