import pickle

# import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import copy

from fast_march_all import PipeLine

# import itk

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging

logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)

import utils  # import get_sitk_image, display_images

#elastic_dir = "/media/sansomk/510808DF6345C808/caseFiles/elastix"
elastic_dir = "/Volumes/SD/caseFiles/elastix"
#elastic_dir = "/media/store/krs/caseFiles/elastix"


trans_params = ['CenterOfRotationPoint', 'CompressResultImage',
                'DefaultPixelValue', 'Direction',
                'FinalBSplineInterpolationOrder', 'FixedImageDimension',
                'FixedInternalImagePixelType', 'HowToCombineTransforms',
                'Index', 'InitialTransformParametersFileName',
                'MovingImageDimension', 'MovingInternalImagePixelType',
                'NumberOfParameters', 'Origin', 'ResampleInterpolator', 
                'Resampler', 'ResultImageFormat', 'ResultImagePixelType',
                'Size', 'Spacing', 'Transform', 'TransformParameters',
                'UseDirectionCosines']

trans_defaults = {'CompressResultImage': ['false'],
                  'FixedImageDimension': ['2'],
                  'FixedInternalImagePixelType': ['float'],
                  'HowToCombineTransforms': ['Compose'],
                  'Index': ['0', '0'],
                  'InitialTransformParametersFileName': ['NoInitialTransform'],
                  'MovingImageDimension': ['2'],
                  'MovingInternalImagePixelType': ['float'],
                  'NumberOfParameters': ['3'],
                  'ResultImageFormat': ['nii'],
                  'ResultImagePixelType': ['float'],
                  'UseDirectionCosines': ['true']
                  }


# This function evaluates the metric value in a thread safe manner
def evaluate_metric_rgb(current_transform, f_sitk, t_sitk, f_mask, t_mask,  n_bins):
  f_image = []
  m_image = []
  select = sitk.VectorIndexSelectionCastImageFilter()
  m_image.append(select.Execute(t_sitk, 0, t_sitk.GetPixelID())*t_mask)
  m_image.append(select.Execute(t_sitk, 1, t_sitk.GetPixelID())*t_mask)
  m_image.append(select.Execute(t_sitk, 2, t_sitk.GetPixelID())*t_mask)

  select2 = sitk.VectorIndexSelectionCastImageFilter()
  f_image.append(select2.Execute(f_sitk, 0, f_sitk.GetPixelID())*f_mask)
  f_image.append(select2.Execute(f_sitk, 1, f_sitk.GetPixelID())*f_mask)
  f_image.append(select2.Execute(f_sitk, 2, f_sitk.GetPixelID())*f_mask)
  registration_method = sitk.ImageRegistrationMethod()
  #registration_method.SetMetricFixedMask(mask)
  #registration_method.SetMetricMovingMask(mask)
  #registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=4)
  registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=n_bins)
  registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
  registration_method.SetMetricSamplingPercentage(0.2)
  registration_method.SetInterpolator(sitk.sitkLinear)
  registration_method.SetInitialTransform(current_transform)
  res = []
  for a, b in zip(f_image, m_image):
    res.append(registration_method.MetricEvaluate(sitk.Cast(a, sitk.sitkFloat32),
                                                  sitk.Cast(b, sitk.sitkFloat32)
                                                  )
              )

  return res


# n_max is the maxmimum picture size for registration
def stage_1b_transform(reg_dict, n_max, initial_transform, count=0):
  # print(fixed[1]["crop_paths"])
  # load color images
  ff_path = reg_dict["f_row"]["color_paths"]
  tf_path = reg_dict["t_row"]["color_paths"]

  ff_mask_path = reg_dict["f_row"]["mask_path"]
  tf_mask_path = reg_dict["t_row"]["mask_path"]

  for page in reg_dict["f_page"]:
      if page["size_x"] > n_max:
          continue
      break
  page_idx = page["index"]

  #print(page_idx, page)
  spacing = (page["mmp_x"], page["mmp_y"])
  #print(spacing)
  # transform numpy array to simpleITK image
  # have set the parameters manually
  im_f = tiff.imread(ff_path, key=page_idx)
  f_sitk = utils.get_sitk_image(im_f[:, :, :3], spacing=spacing, vector=True)

  im_t = tiff.imread(tf_path, key=page_idx)
  t_sitk = utils.get_sitk_image(im_t[:, :, :3], spacing=spacing, vector=True)

  im_mask_f = sitk.ReadImage(ff_mask_path)
  im_mask_t = sitk.ReadImage(tf_mask_path)

  
  identity = sitk.Transform(im_mask_f.GetDimension(), sitk.sitkIdentity)
  f_mask_resampled = sitk.Resample(im_mask_f, f_sitk,
                                    identity, sitk.sitkNearestNeighbor,
                                    0.0, im_mask_f.GetPixelID())
  t_mask_resampled = sitk.Resample(im_mask_t, t_sitk,
                                    identity, #initial_transform["transform"],
                                    sitk.sitkNearestNeighbor,
                                    0.0, im_mask_t.GetPixelID())

  #TODO move to a function
  negated = (sitk.GetArrayViewFromImage(f_mask_resampled) + 1) % 2
  im_test = sitk.GetArrayViewFromImage(t_sitk)
  print(t_sitk.GetOrigin())
  #luminance of background
  test = (0.3 * im_test[:,:,0] * negated +
          0.59 * im_test[:,:,1] * negated +
          0.11 * im_test[:,:,2] * negated)		
  mean = test[test > 0].mean()

  moving_resampled = sitk.Resample(t_sitk, f_sitk,
                                   initial_transform["transform"],
                                   sitk.sitkLinear, round(mean), t_sitk.GetPixelID())
  #new_image = utils.resample_rgb(composite, f_sitk, t_sitk, mean=mean)
  checkerboard = sitk.CheckerBoardImageFilter()
  check_im = checkerboard.Execute(f_sitk, moving_resampled, (8,8))

  # new_array = np.empty_like(im_f[:, :, :3])
  # new_array[:, :, 0] = 
  # new_array[:, :, 1] = sitk.GetArrayViewFromImage(t_resampled1)
  # new_array[:, :, 2] = sitk.GetArrayViewFromImage(t_resampled2)
  new_fig = utils.display_images(
      fixed_npa=sitk.GetArrayViewFromImage(f_sitk),
      moving_npa=sitk.GetArrayViewFromImage(moving_resampled),
      checkerboard=sitk.GetArrayViewFromImage(check_im),
      show=False
  )
  #
  
  #cnt_pixels = np.count_nonzero(sitk.GetArrayViewFromImage(mask_resampled))
  n_bins = int(np.cbrt(np.prod(sitk.GetArrayViewFromImage(f_mask_resampled).shape)))

  metric = evaluate_metric_rgb(initial_transform["transform"], f_sitk, t_sitk, f_mask_resampled, t_mask_resampled, n_bins)
  # get the root mean sum squared
  rmse = -np.sqrt(1.0/3.0*(metric[0]**2.0 + metric[1]**2.0 + metric[2]**2.0))
  #print(rmse)
  best_reg = dict(
                  transform = initial_transform["transform"],
                  measure = rmse,
                  tiff_page = page
                  )

  return best_reg, new_fig

