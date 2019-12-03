import pickle 
#import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import os
import copy
#import itk

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

import utils #import get_sitk_image, display_images


# n_max is the maxmimum picture size for registration
def stage_1_transform(reg_dict, n_max, init_params, count=0):
  #print(fixed[1]["crop_paths"])
  ff_path = reg_dict["f_row"]["crop_paths"]
  tf_path = reg_dict["t_row"]["crop_paths"]

  ff_mask_path = reg_dict["f_row"]["mask_path"]
  tf_mask_path = reg_dict["t_row"]["mask_path"]

  #base_res_x = reg_dict["f_row"]["mpp-x"]
  #base_res_y = reg_dict["f_row"]["mpp-y"]

  for page in reg_dict["f_page"]:
    if( page["size_x"] > n_max):
      continue
    break
  page_idx = page["index"]

  #print(page_idx, page)
  spacing = ( page["mmp_x"], page["mmp_y"] )
  #print(spacing)
  # transform numpy array to simpleITK image
  # have set the parameters manually
  im_f = tiff.imread(ff_path, key=page_idx)
  f_sitk = utils.get_sitk_image(im_f, spacing)

  im_t = tiff.imread(tf_path, key=page_idx)
  t_sitk = utils.get_sitk_image(im_t, spacing)

  im_mask_f = sitk.ReadImage(ff_mask_path)
  im_mask_t = sitk.ReadImage(tf_mask_path)

  # this is the union operator take the maximum of the two images
  #max_mask = sitk.Maximum(im_mask_f, im_mask_t)
  identity = sitk.Transform(im_mask_f.GetDimension(), sitk.sitkIdentity)
  f_mask_resampled = sitk.Resample(im_mask_f, f_sitk,
                                    identity, sitk.sitkNearestNeighbor,
                                    0.0, im_mask_f.GetPixelID())
  t_mask_resampled = sitk.Resample(im_mask_t, t_sitk,
                                    identity, sitk.sitkNearestNeighbor,
                                    0.0, im_mask_t.GetPixelID())

  f_input = sitk.Cast(f_sitk * f_mask_resampled, sitk.sitkFloat32)
  t_input = sitk.Cast(t_sitk * t_mask_resampled, sitk.sitkFloat32)
  initial_transform = sitk.CenteredTransformInitializer(f_input, 
                                                    t_input, 
                                                    sitk.Euler2DTransform(), 
                                                    sitk.CenteredTransformInitializerFilter.MOMENTS)
  #print(initial_transform)
  #trans = initial_transform.GetParameters()
  #center = initial_transform.GetFixedParameters()

  # pick a narrower region to attempt registration
  # take the best guess and then get  a range of
  # angles that span the plus or minus 5/7 of one "tick"
  # in rotation from the guess
  best_init = init_params["best_angle"]
  angle_range = (5.0 / 7.0) * 2.0 * np.pi / init_params["n_angles"]
  n_angles = np.linspace(best_init-angle_range, best_init+angle_range, 5)
  #print(n_angles)
  best_reg = {}
  # this is where we could put a loop to iterate over the rotation angle
  rot = sitk.Euler2DTransform(initial_transform)
  #print(rot)
  #quit()
  #rot.SetCenter(center)

  reg_method = sitk.ImageRegistrationMethod()
  #reg_method.SetMetricFixedMask(mask_resampled)
  #reg_method.SetMetricMovingMask(mask_resampled)

  #cnt_pixels = np.count_nonzero(sitk.GetArrayViewFromImage(mask_resampled))
  n_bins = int(np.cbrt(np.prod(sitk.GetArrayViewFromImage(f_mask_resampled).shape)))
  # Similarity metric settings.
  reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=n_bins)
  #reg_method.SetMetricAsANTSNeighborhoodCorrelation(radius=4)
  reg_method.SetMetricSamplingStrategy(reg_method.RANDOM) #REGULAR
  reg_method.SetMetricSamplingPercentage(0.2)
  reg_method.SetInterpolator(sitk.sitkLinear)

  # Optimizer settings.
  reg_method.SetOptimizerAsGradientDescent(learningRate=0.8,
                                          numberOfIterations=1000,
                                          convergenceMinimumValue=1e-10,
                                          convergenceWindowSize=20)
  reg_method.SetOptimizerScalesFromPhysicalShift()

  # Setup for the multi-resolution framework.            
  reg_method.SetShrinkFactorsPerLevel(shrinkFactors =     [ 8, 8, 4, 4, 2, 2, 1, 1])
  # reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [3.0, 2.72727273, 2.45454545, 2.18181818, 
  #                                                         1.90909091, 1.63636364, 1.36363636, 
  #                                                         1.09090909, 0.81818182, 0.54545455, 
  #                                                         0.27272727, 0.0])
  # reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [1.96      , 1.78181818, 1.60363636, 1.42545455, 1.24727273,
  #                                                         1.06909091, 0.89090909, 0.71272727, 0.53454545, 0.35636364,
  #                                                         0.17818182, 0.        ])
  #reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2., 1., 0., 2., 1., 0., 2., 1., 0., 2., 1., 0.])
  #reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0])
  reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [0.0, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0])
  #reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
  reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

  # Connect all of the observers so that we can perform plotting during registration.
  # reg_method.AddCommand(sitk.sitkStartEvent, utils.start_plot)
  # reg_method.AddCommand(sitk.sitkEndEvent, lambda: utils.end_plot(angle))
  # reg_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, utils.update_multires_iterations) 
  # reg_method.AddCommand(sitk.sitkIterationEvent, lambda: utils.plot_values(reg_method))

  min_metric = 9999999.0
  for angle in n_angles:
    rot.SetAngle(angle)
    #print(initial_transform)
    exception = True
    count_2 = 0
    reg_method.SetInitialTransform(rot, inPlace=False)
    while (exception == True) and (count_2 < 20):
      try:
        final_transform = reg_method.Execute(f_input, 
                                             t_input)
      except RuntimeError as e:
        count_2 += 1
        print("Got an exception\n" + str(e))
        continue
      exception = False


    measure = reg_method.GetMetricValue()
    if ( measure < min_metric):
      min_metric = copy.deepcopy(measure)
      best_reg = dict( angle = angle,
                      transform = final_transform,
                      measure = measure,
                      stop_cond = reg_method.GetOptimizerStopConditionDescription(),
                      tiff_page = page # this contains the page from the tiff file
                      )
    print(measure, reg_method.GetOptimizerStopConditionDescription())
    # t_resampled = sitk.Resample(t_sitk, f_sitk,
    #                              final_transform, sitk.sitkLinear,
    #                              0.0, t_sitk.GetPixelID())
    # utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
    #            moving_npa = sitk.GetArrayViewFromImage(t_resampled))
    
    # utils.display_image(sitk.GetArrayFromImage(f_sitk*f_mask_resampled))
    # utils.display_image(sitk.GetArrayFromImage(t_sitk*t_mask_resampled))

    # f_rescale = sitk.RescaleIntensityImageFilter()
    # f_rescale.SetOutputMaximum(1)
    # f_rescale.SetOutputMinimum(0)
  
  #2,(reg_dict["f_page"][0]["scale_x"], reg_dict["f_page"][0]["scale_y"]))
  print('Final metric value: {0}'.format(best_reg["measure"]))
  print('Optimizer\'s stopping condition, {0}'.format(best_reg["stop_cond"]))

  moving_resampled = sitk.Resample(t_sitk, f_sitk,
                                   best_reg["transform"], sitk.sitkLinear,
                                   0.0, t_sitk.GetPixelID())

  checkerboard = sitk.CheckerBoardImageFilter()
  check_im = checkerboard.Execute(f_sitk, moving_resampled, (8,8))

  fig_stage1 = utils.display_images(
      fixed_npa=sitk.GetArrayViewFromImage(f_sitk),
      moving_npa=sitk.GetArrayViewFromImage(moving_resampled),
      checkerboard=sitk.GetArrayViewFromImage(check_im),
      show=False
  )

  
  #print(stuff)
  #stuff = {}
  #return stuff
  while (best_reg["measure"] > -0.40):
    # try up to two more times
    if(count < 2):
      count += 1
      plt.close(fig_stage1)
      best_reg, fig_stage1 = stage_1_transform(reg_dict, n_max, init_params, count)
    else:
      break
    # else:
    #     #basically this isn't goog enough so try at higher resolution
    #     new_max = n_max*2.0
    #     if (new_max <= (reg_dict["f_page"][0]['size_x'] // 8 ) ):
    #         print("Increasing the image resolution to {0}".format(new_max))
    #         best_reg = stage_1_transform(reg_dict, new_max, init_params, 0)
    #     else:
    #         print("I have run out of resolution")
    #         break

  return best_reg, fig_stage1




class register_series:

  def __init__(self):
    #self.trans_image_filter = sitk.ElastixImageFilter()
    self.trans_image_filter = sitk.TransformixImageFilter()
    self.trans_param_map = sitk.GetDefaultParameterMap("rigid")
    #self.trans_param_map = sitk.GetDefaultParameterMap('translation')
    #self.trans_param_map = sitk.GetDefaultParameterMap('affine')

    
    self.trans_param_map["WriteIterationInfo"] = ['true']
    self.trans_param_map["MaximumNumberOfIterations"] = ["512"]
    self.trans_param_map["WriteResultImage"] = ['false']
    self.trans_param_map["ResampleInterpolator"] = ['LinearResampleInterpolator']
    self.trans_param_map["Origin"] = ["0.0", "0.0"]
    self.trans_param_map["Origin"] = ["0.0", "0.0"]
    self.trans_param_map["Direction"] = ["1.0", "0.0", "0.0", "1.0"]
    
    #print(sitk.PrintParameterMap(self.trans_param_map))

    self.trans_image_filter.SetTransformParameterMap(self.trans_param_map)
    self.moving = None
    self.cur_file_name = None
    self.new_path = None
    self.cur_data = None
    self.index = 0
  
  def update_image(self, row_data):
    #print(row_data)
    self.cur_file_name = row_data["crop_paths"]
    self.cur_data = row_data.to_dict()
    print(self.cur_data )
    self.moving = sitk.ReadImage(self.cur_file_name, sitk.sitkInt16)
    sz = self.moving.GetSize()
    print(self.moving.GetDirection())
    # thickness = str(self.cur_data["slice_thickness"] + self.cur_data["gap"])
    spacing = (self.cur_data["mpp-x"], self.cur_data["mpp-y"])
    self.trans_param_map["Spacing"] = [str(spacing[0]), str(spacing[1])]
    self.trans_param_map["Size"] = [str(sz[0]), str(sz[1])]
    self.trans_param_map["Index"] = [str(self.index)]
    self.trans_image_filter.SetTransformParameterMap(self.trans_param_map)

    self.moving.SetSpacing(spacing)
    self.moving.SetOrigin((0.0,0.0))
    self.trans_image_filter.SetMovingImage(self.moving)
    self.index += 1

  def series_update(self, filename):
    image = sitk.ReadImage(filename, sitk.sitkInt16)
    self.trans_image_filter.SetMovingImage(image)

  def Execute(self):
    self.trans_image_filter.Execute()
  
  def from_list_of_paths(self, file_list):
    for f in file_list:
      self.series_update(f)
      self.Execute()

  def write_new_file(self):
    split = os.path.splitext(self.cur_file_name)
    self.new_path = split[0] + "_result" + split[1]
    sitk.WriteImage(self.trans_image_filter.GetResultImage(), self.new_path)
