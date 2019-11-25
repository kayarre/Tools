import pickle

# import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk

# import itk

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging

logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)

import utils  # import get_sitk_image, display_images

#elastic_dir = "/media/sansomk/510808DF6345C808/caseFiles/elastix"
#elastic_dir = "/Volumes/SD/caseFiles/elastix"
elastic_dir = "/media/store/krs/caseFiles/elastix"



# This function evaluates the metric value in a thread safe manner
def evaluate_metric_rgb(current_transform, f_sitk, t_sitk):
  f_image = []
  m_image = []
  select = sitk.VectorIndexSelectionCastImageFilter()
  m_image.append(select.Execute(t_sitk, 0, t_sitk.GetPixelID()))
  m_image.append(select.Execute(t_sitk, 1, t_sitk.GetPixelID()))
  m_image.append(select.Execute(t_sitk, 2, t_sitk.GetPixelID()))

  select2 = sitk.VectorIndexSelectionCastImageFilter()
  f_image.append(select2.Execute(f_sitk, 0, f_sitk.GetPixelID()))
  f_image.append(select2.Execute(f_sitk, 1, f_sitk.GetPixelID()))
  f_image.append(select2.Execute(f_sitk, 2, f_sitk.GetPixelID()))
  registration_method = sitk.ImageRegistrationMethod()
  #registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=4)
  registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
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

  # base_res_x = reg_dict["f_row"]["mpp-x"]
  # base_res_y = reg_dict["f_row"]["mpp-y"]

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
  # f_sitk = utils.get_sitk_image(im_f, spacing)
  f_sitk = utils.get_sitk_image(im_f[:, :, :3], spacing=spacing, vector=True)

  im_t = tiff.imread(tf_path, key=page_idx)
  # t_sitk = utils.get_sitk_image(im_t, spacing)
  t_sitk = utils.get_sitk_image(im_t[:, :, :3], spacing=spacing, vector=True)

  numberOfChannels = 3
  elastix = sitk.ElastixImageFilter()
  for idx in range(numberOfChannels):
    elastix.AddFixedImage(sitk.VectorIndexSelectionCast(f_sitk, idx))
    elastix.AddMovingImage(sitk.VectorIndexSelectionCast(t_sitk, idx))

  #print("number of images:")
  #print(elastix.GetNumberOfFixedImages())
  #print(elastix.GetNumberOfMovingImages())

  mean = utils.get_mean_edges(t_sitk)
  # read parameter file from disk so we are using the same file as command line
  rigid = sitk.GetDefaultParameterMap("rigid")
  rigid["DefaultPixelValue"] = [str(mean)]
  rigid["WriteResultImage"] = ["false"]
  #rigid["Metric"] = ["AdvancedNormalizedCorrelation"]
  #rigid[ "MaximumNumberOfIterations" ] = [ "1" ]
  # sitk.PrintParameterMap(rigid)

  
  transform_init = initial_transform["transform"]
  #print(transform_init)
  in_params = list(transform_init.GetParameters())
  fixed_in_params = transform_init.GetFixedParameters()
  rigid["TransformParameters"] = [str(a) for a in in_params]
  rigid["CenterOfRotationPoint"] = [str(a) for a in fixed_in_params]

  rigid_in_params = {}
  for key, value in rigid.items():
    rigid_in_params[key] = value
    # print(rigid["TransformParameters"])

  affine = sitk.GetDefaultParameterMap("affine")
  affine["NumberOfResolutions"] = ["6"]
  affine["DefaultPixelValue"] = [str(mean)]
  affine["WriteResultImage"] = ["false"]
  # affine["Registration"] =  ["MultiResolutionRegistrationWithFeatures"]
  # affine["Metric"] =  ["KNNGraphAlphaMutualInformation"]
  # affine["ImageSampler"] =  ["MultiInputRandomCoordinate"]
  # # KNN specific
  # affine["Alpha"] =  ["0.99"]
  # affine["AvoidDivisionBy"] =  ["0.0000000001"]
  # affine["TreeType"] =  ["KDTree"]
  # affine["BucketSize"] =  ["50"]
  # affine["SplittingRule"] =  ["ANN_KD_STD"]
  # affine["ShrinkingRule"] =  ["ANN_BD_SIMPLE"]
  # affine["TreeSearchType"] =  ["Standard"]
  # affine["KNearestNeighbours"] =  ["20"]
  # affine["ErrorBound"] =  ["10"]
  #affine["FixedImagePyramid"]

  # (FixedImagePyramid "FixedSmoothingImagePyramid" "FixedSmoothingImagePyramid")
  # (MovingImagePyramid "MovingSmoothingImagePyramid" "MovingSmoothingImagePyramid")
  #(Interpolator "BSplineInterpolator" "BSplineInterpolator")
  #affine["Metric"] = ["AdvancedNormalizedCorrelation"]
  #affine[ "MaximumNumberOfIterations" ] = [ "1" ]

  parameterMapVector = sitk.VectorOfParameterMap()
  parameterMapVector.append(rigid)
  parameterMapVector.append(affine)

  #print(sitk.PrintParameterMap(affine))

  elastix.SetParameterMap(parameterMapVector)

  elastix.SetOutputDirectory(elastic_dir)
  elastix.LogToConsoleOff()
  elastix.SetLogToFile(True)

  exception = True
  count = 0
  while (exception == True) and (count < 20):
    try:
      # Don't optimize in-place, we would possibly like to run this cell multiple times.
      elastix.Execute()
      #print("was here")
    except RuntimeError as e:
      count += 1
      print("Got an exception\n" + str(e))
      continue
    exception = False

  # moving_resampled = elastix.GetResultImage()
  #print(elastix)

  #print(dir(elastix))
  tran_map = elastix.GetTransformParameterMap()
  # print(tran_map)
  # print("\n")
  # print(sitk.PrintParameterMap(tran_map))
  trans_out_params = {}
  # print(len(tran_map))
  for idx, t in enumerate(tran_map):
    trans = {}
    for key, value in t.items():
        trans[key] = value
    trans_out_params[idx] = trans

  # print(rigid_in_params)
  # print("\n")
  # print(trans_out_params)

  # print(tran_map.GetTransformParameter("CenterOfRotationPoint"))
  # print(tran_map["TransformParameters"])
  # test_image = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 0), tran_map)
  # test_image1 = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 1), tran_map)
  # test_image2 = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 2), tran_map)

  # new_array = np.empty_like(im_f[:, :, :3])
  # new_array[:, :, 0] = sitk.GetArrayViewFromImage(test_image)
  # new_array[:, :, 1] = sitk.GetArrayViewFromImage(test_image1)
  # new_array[:, :, 2] = sitk.GetArrayViewFromImage(test_image2)
  # utils.display_images(
  #     fixed_npa=sitk.GetArrayViewFromImage(f_sitk), moving_npa=new_array
  # )

  # this seemed like the correct approach but is it not.
  #composite = sitk.Transform(transform_init)
  # this appears to work and give better results
  composite = sitk.Transform(2, sitk.sitkIdentity)
  #print(composite)
  for idx, trans in trans_out_params.items():
    trans_type = trans["Transform"][0]
    # in_params = list(trans.GetParameters())
    # fixed_in_params = trans.GetFixedParameters()
    im_dim = int(trans["FixedImageDimension"][0])

    CenterOfRotationPoint = [float(a) for a in trans["CenterOfRotationPoint"]]
    TransformParameters = [float(a) for a in trans["TransformParameters"]]

    if ( (trans_type == "EulerTransform") and (im_dim == 2)):
      trans_add = sitk.Euler2DTransform()
      if ("ComputeZYX" in trans.keys()):
        computeZYX = trans["ComputeZYX"]
      else:
        computeZYX = False
      trans_add.SetCenter(CenterOfRotationPoint)
      #trans_add.SetTranslation()
      trans_add.SetAngle(TransformParameters[0])
      trans_add.SetTranslation(TransformParameters[1:])
    elif ( (trans_type == "AffineTransform") and (im_dim == 2)):
      #similarity_trans = sitk.Similarity2D
      trans_add = sitk.AffineTransform(im_dim)
      if ("ComputeZYX" in trans.keys()):
        computeZYX = trans["ComputeZYX"]
      else:
        computeZYX = False

      trans_add.SetCenter(CenterOfRotationPoint)
      trans_add.SetMatrix(TransformParameters[0:4])
      trans_add.SetTranslation(TransformParameters[4:])
    else:
      print(" can't convert this type of transform")

    #print(trans_add)
    #quit()
    composite.AddTransform(trans_add)
    #[rx,ry,rz] = 
    # rigid["TransformParameters"] = [str(a) for a in in_params]
    # rigid["CenterOfRotationPoint"] = [str(a) for a in fixed_in_params]
  #print(composite)

  mean = utils.get_mean_edges(t_sitk)
  moving_resampled = sitk.Resample(t_sitk, f_sitk,  composite, sitk.sitkLinear, mean, t_sitk.GetPixelID())
  #new_image = utils.resample_rgb(composite, f_sitk, t_sitk, mean=mean)
  new_image = moving_resampled
  checkerboard = sitk.CheckerBoardImageFilter()
  check_im = checkerboard.Execute(f_sitk, moving_resampled)#new_image)

  # new_array = np.empty_like(im_f[:, :, :3])
  # new_array[:, :, 0] = 
  # new_array[:, :, 1] = sitk.GetArrayViewFromImage(t_resampled1)
  # new_array[:, :, 2] = sitk.GetArrayViewFromImage(t_resampled2)
  new_fig = utils.display_images(
      fixed_npa=sitk.GetArrayViewFromImage(f_sitk),
      moving_npa=sitk.GetArrayViewFromImage(new_image),
      checkerboard=sitk.GetArrayViewFromImage(check_im),
      show=False
  )

  metric = evaluate_metric_rgb(composite, t_sitk, f_sitk)
  # get the root mean sum squared
  rmse = -np.sqrt(1.0/3.0*(metric[0]**2.0 + metric[1]**2.0 + metric[2]**2.0))
  #print(rmse)
  best_reg = dict(
                  transform = composite,
                  measure = rmse,
                  tiff_page = page
                  )

  return best_reg, new_fig

