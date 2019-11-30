#!/usr/bin/env python

import SimpleITK as sitk
#import sys
import os
import utils
import tifffile as tiff
import utils
import itk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# HAVE_NUMPY = True
# try:
#   import numpy
# except ImportError:
#   HAVE_NUMPY = False


# def _get_itk_pixelid(numpy_array_type):
#     """Returns a ITK PixelID given a numpy array."""

#     if not HAVE_NUMPY:
#         raise ImportError('Numpy not available.')
#     import itk
#     # This is a Mapping from numpy array types to itk pixel types.
#     _np_itk = {numpy.uint8:itk.UC,
#                 numpy.uint16:itk.US,
#                 numpy.uint32:itk.UI,
#                 numpy.uint64:itk.UL,
#                 numpy.int8:itk.SC,
#                 numpy.int16:itk.SS,
#                 numpy.int32:itk.SI,
#                 numpy.int64:itk.SL,
#                 numpy.float32:itk.F,
#                 numpy.float64:itk.D,
#                 numpy.complex64:itk.complex[itk.F],
#                 numpy.complex128:itk.complex[itk.D]
#                 }
#     try:
#         return _np_itk[numpy_array_type.dtype.type]
#     except KeyError as e:
#         for key in _np_itk:
#             if numpy.issubdtype(numpy_array_type.dtype.type, key):
#                 return _np_itk[key]
#             raise e

# def _GetImageFromArray(arr, function, is_vector):
#   """Get an ITK image from a Python array.
#   """
#   if not HAVE_NUMPY:
#     raise ImportError('Numpy not available.')
#   import itk
#   PixelType = _get_itk_pixelid(arr)
#   if is_vector:
#     Dimension = arr.ndim - 1
#     if arr.flags['C_CONTIGUOUS']:
#       VectorDimension = arr.shape[-1]
#     else:
#       VectorDimension = arr.shape[0]
#     if PixelType == itk.UC:
#       if VectorDimension == 3:
#         ImageType = itk.Image[ itk.RGBPixel[itk.UC], Dimension ]
#       elif VectorDimension == 4:
#         ImageType = itk.Image[ itk.RGBAPixel[itk.UC], Dimension ]
#     else:
#       ImageType = itk.Image[ itk.Vector[PixelType, VectorDimension] , Dimension]
#   else:
#     Dimension = arr.ndim
#     ImageType = itk.Image[PixelType, Dimension]
#   templatedFunction = getattr(itk.PyBuffer[ImageType], function)
#   return templatedFunction(arr, is_vector)



class PipeLine:
  """A simple PipeLine class"""

  def __init__(self):
    self.smooth = sitk.CurvatureAnisotropicDiffusionImageFilter()
    self.smooth.SetTimeStep(0.125)
    #self.smooth.SetTimeStep(0.0625)
    self.smooth.SetNumberOfIterations(5)
    self.smooth.SetConductanceParameter(9.0)
    self.smooth.GlobalWarningDisplayOff()

    self.rescaleUIint8 = sitk.RescaleIntensityImageFilter()
    self.rescaleUIint8.SetOutputMinimum(0.0)
    self.rescaleUIint8.SetOutputMaximum(254.0)

    self.rescale_1 = sitk.RescaleIntensityImageFilter()
    self.rescale_1.SetOutputMinimum(0.0)
    self.rescale_1.SetOutputMaximum(1.0)

    self.rescale_speed = sitk.RescaleIntensityImageFilter()
    self.rescale_speed.SetOutputMinimum(-1.0)
    self.rescale_speed.SetOutputMaximum(1.0)

    self.sigmoid = sitk.SigmoidImageFilter()
    self.sigmoid.SetOutputMinimum(0.0)
    self.sigmoid.SetOutputMaximum(1.0)
    # self.sigmoid.SetBeta(32)#(0.2) defines the intensity around which the range is centered
    # # alpha defines the width of the input intensity range
    # self.sigmoid.SetAlpha(10)#(0.1)# Create an itk image from the simpleitk image via numpy array

    self.sigmoid.SetBeta(64)#(0.2) defines the intensity around which the range is centered
    # alpha defines the width of the input intensity range
    self.sigmoid.SetAlpha(32)#(0.1)# Create an itk image from the simpleitk image via numpy array


    self.gradmag = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    self.sigma = 4.0
    self.gradmag.SetSigma(self.sigma)

    self.march = sitk.FastMarchingUpwindGradientImageFilter()
    #self.march.NormalizeAcrossScaleOn()
    #self.march.SetSigma(2.0)
    #self.march.DebugOn()
    #self.march.GlobalDefaultDebugOn()

    self.march2 = sitk.FastMarchingImageFilter()

    self.lsFilter = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
    self.lsFilter.UseImageSpacingOn()
    self.lsFilter.SetMaximumRMSError(0.0)
    self.lsFilter.SetNumberOfIterations(500)
    self.lsFilter.SetLambda1(1) # Weight for internal levelset term contribution to the total energy.
    self.lsFilter.SetLambda2(1) # Weight for external levelset term contribution to the total energy. 
    self.lsFilter.SetEpsilon(1.0) # upsampling
    self.lsFilter.SetCurvatureWeight(1.0) # Weight for curvature. Higher results in smoother levelsets, but less ability to capture fine features.
    self.lsFilter.SetAreaWeight(0.0)
    self.lsFilter.SetReinitializationSmoothingWeight(0.0)
    self.lsFilter.SetVolume(0.0)
    self.lsFilter.SetVolumeMatchingWeight(0.0)
    self.lsFilter.SetHeavisideStepFunction(self.lsFilter.AtanRegularizedHeaviside)

    self.binary_fill = sitk.BinaryFillholeImageFilter()
    self.binary_fill.FullyConnectedOff()
    self.binary_fill.SetForegroundValue(0.0)

    self.dilate = sitk.BinaryDilateImageFilter()
    self.dilate.BoundaryToForegroundOff()
    self.dilate.SetKernelType(sitk.sitkBall)
    self.dilate.SetKernelRadius(1)
    self.dilate.SetForegroundValue(0)
    self.dilate.SetBackgroundValue(1)

    self.erode = sitk.BinaryErodeImageFilter()
    self.erode.BoundaryToForegroundOff()
    self.erode.SetKernelType(sitk.sitkBall)
    self.erode.SetKernelRadius(1)
    self.erode.SetForegroundValue(0)
    self.erode.SetBackgroundValue(1)

    self.mask_neg = sitk.MaskNegatedImageFilter()
    #self.mask_neg = sitk.MaskImageFilter()
    #self.mask_neg.SetMaskingValue(255)
    self.mask_neg.SetOutsideValue(0)

    self.mask = sitk.MaskImageFilter()
    self.mask.SetOutsideValue(0)

    self.checkerboard = sitk.CheckerBoardImageFilter()

    self.thresh = sitk.ThresholdImageFilter() #BinaryThresholdImageFilter()
    #self.thresh.ThresholdBelow(10)
    #self.thresh.SetUpper(10)
    #self.thresh.SetUpperThreshold(10)
    #self.thresh.SetOutsideValue(0)
    #self.thresh.SetInsideValue(0)

    self.writer = sitk.ImageFileWriter()

    self.signedmaurer = sitk.SignedMaurerDistanceMapImageFilter()
    self.signedmaurer.SquaredDistanceOff()
    self.signedmaurer.UseImageSpacingOn()
    self.signedmaurer.InsideIsPositiveOff()

    self.cc = sitk.ConnectedComponentImageFilter()
    self.cc.FullyConnectedOff()

    self.label_stats = sitk.LabelIntensityStatisticsImageFilter()
    self.stats = sitk.StatisticsImageFilter()

    self.binary_opening = sitk.BinaryOpeningByReconstructionImageFilter()
    self.binary_opening.FullyConnectedOn()
    self.binary_opening.SetKernelType(sitk.sitkBall)
    self.binary_opening.SetKernelRadius(16)
    self.binary_opening.SetBackgroundValue(1.0)
    self.binary_opening.SetForegroundValue(0.0)

    self.reciprocal = sitk.BoundedReciprocalImageFilter()

    self.gauss = sitk.RecursiveGaussianImageFilter()
    self.gauss.NormalizeAcrossScaleOn()
    self.gauss.SetSigma(4.0)
    self.gauss.SetOrder(0)

    self.normalize = sitk.NormalizeImageFilter()

    self.median = sitk.MedianImageFilter()

    self.resample_nn = sitk.ResampleImageFilter() 
    self.resample_nn.SetInterpolator(sitk.sitkNearestNeighbor)


def main():
  # Create Masks for the images

  case_file = "case_1.pkl"
  csv_file = "case_1.csv"
  #n_max = 256
  n_max = 128

  top_dir = "/Volumes/SD/caseFiles"
  #top_dir = "/media/store/krs/caseFiles"
  # top_dir = "/media/sansomk/510808DF6345C808/caseFiles"
  pickle_path = os.path.join(top_dir, case_file)
  csv_path = os.path.join(top_dir, csv_file)
  df = pd.read_pickle(pickle_path)

  mask_dir = "masks"
  images_dir = "images"
  images_path = os.path.join(top_dir, images_dir)

  mask_dir_path = os.path.join(top_dir, mask_dir)

  mask_path_list = []
  mask_name_list = []

  pipe = PipeLine()
  display_extra = False
  for idx, pd_data in df.iterrows():
    # if (idx in [0, 1, 14, 35, 37, 38 ]):
    #   display_extra = True
    # else:
    #   display_extra = False
    #   continue
    # if ( idx != 14):
    #   continue
    tiff_path = pd_data["crop_paths"] # ["color_paths"]
    page_data = utils.get_additional_info(pd_data)

    for page in page_data:
        if( page["size_x"] > n_max):
            continue
        break
    page_idx = page["index"]
    #print(page)
    #quit()
    spacing = ( page["mmp_x"], page["mmp_y"] )
    print(spacing)
    in_image = tiff.imread(tiff_path, key=page_idx)

    #print(in_image.shape, in_image.dtype)
    #quit()

    # use corners for seeds
    shape = in_image.shape
    seedValue = 0
    #trialPoint = (seedPosition[0], seedPosition[1], seedValue)
    seedPositions = []
    seedPositions.append((int(0),   int(0),   seedValue))
    seedPositions.append((int(0),   int(shape[1]-1), seedValue))
    seedPositions.append((int(shape[0]-1), int(0),   seedValue))
    seedPositions.append((int(shape[0]-1), int(shape[1]-1), seedValue))

    # Create an itk image from the simpleitk image via numpy array
    # itk_image = itk.GetImageFromArray(in_image, is_vector = False)
    # itk_image.SetOrigin((0.0,0.0))
    # itk_image.SetSpacing(spacing)   
    # new_direction = numpy.eye(2)
    # itk_image.SetDirection(itk.matrix_from_array(new_direction))

    # grayscale_filter = itk.RGBToLuminanceImageFilter.New(itk_image)
    # grayscale_filter.Update()
    # grayscale = grayscale_filter.GetOutput()

    # Back to a simpleitk image from the itk image
    # new_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(grayscale), 
    #     isVector = grayscale.GetNumberOfComponentsPerPixel()>1)
    # new_sitk_image.SetOrigin(tuple(grayscale.GetOrigin()))
    # new_sitk_image.SetSpacing(tuple(grayscale.GetSpacing()))
    # new_sitk_image.SetDirection(itk.GetArrayFromMatrix(grayscale.GetDirection()).flatten())
    new_sitk_image = utils.get_sitk_image(in_image, spacing)
    inputImage = sitk.Cast(new_sitk_image, sitk.sitkFloat64)
    #shift_im = sitk.ShiftScale(	inputImage, shift = 128, scale = 1.0 )	
    #rescale_in = pipe.rescaleUIint8.Execute(shift_im)
    #utils.display_image(sitk.GetArrayFromImage(rescale_in))

    # dumb pixels
    thresh = pipe.thresh.Execute(inputImage, 20, 10000, 0)

    time_step = 0.8 * pipe.smooth.EstimateOptimalTimeStep(thresh) #0.8*(spacing[0] / (2.0**3.0))
    #print(time_step)
    pipe.smooth.SetTimeStep(time_step)
    smoothingOutput = pipe.smooth.Execute(thresh)

    smooth2 = pipe.gauss.Execute(smoothingOutput)

    grad_mag = pipe.gradmag.Execute(smooth2)
    smooth3 = pipe.gauss.Execute(grad_mag)
    norm_1 = pipe.normalize.Execute(smooth3)

    pipe.march.ClearTrialPoints()
    for pt in seedPositions:
      pipe.march.AddTrialPoint(pt)
    fastMarchingOutput = pipe.march.Execute(norm_1)
    pipe.stats.Execute(fastMarchingOutput)
    median = np.median(sitk.GetArrayFromImage(fastMarchingOutput))

    if (abs(pipe.stats.GetMean() - median) > pipe.stats.GetSigma()):
      print("may need to check your image data")
    # six standard deviations should be good enough
    peak_value = pipe.stats.GetMean() + pipe.stats.GetSigma() * 6.0
    #print(pipe.stats.GetMean(), pipe.stats.GetMaximum(), pipe.stats.GetSigma())
    thresh2 = pipe.thresh.Execute(fastMarchingOutput, 0.0, peak_value, peak_value)
    
    rescale_fm = pipe.rescale_speed.Execute(thresh2)
    if (display_extra):
      utils.display_image(sitk.GetArrayFromImage(rescale_fm))

    output = pipe.lsFilter.Execute(rescale_fm, smoothingOutput)

    # pipe.stats.Execute(output)
    # median = np.median(sitk.GetArrayFromImage(output))
    # if (abs(pipe.stats.GetMean() - median) > pipe.stats.GetSigma()):
    #   print("may need to check your image data")
    # # six standard deviations should be good enough
    # peak_value = pipe.stats.GetMean() + pipe.stats.GetSigma() * 6.0
    #print(pipe.stats.GetMean(), pipe.stats.GetMaximum(), pipe.stats.GetSigma())

    rescale_ = sitk.Cast(pipe.rescaleUIint8.Execute(output), sitk.sitkUInt8) 
   
    binary_out = pipe.binary_fill.Execute(rescale_)

    remove_stuff = pipe.binary_opening.Execute(binary_out)
    if (display_extra):
      utils.display_image(sitk.GetArrayFromImage(remove_stuff))

    pipe.erode.SetKernelRadius(2)
    erode_im = pipe.erode.Execute(remove_stuff) #binary_out)

    pipe.dilate.SetKernelRadius(4)
    dilate_im = pipe.dilate.Execute(erode_im)#remove_stuff)

    mask = sitk.InvertIntensity(dilate_im) - 1

    masked_im = pipe.mask.Execute(sitk.Cast(inputImage, sitk.sitkUInt8), mask)

    #utils.display_image(sitk.GetArrayFromImage(masked_im))

    mask_name = case_file.split(".")[0] + "_mask_{0:04d}.nrrd".format(idx)
    mask_path = os.path.join(mask_dir_path, mask_name)

    mask_name_list.append(mask_name)
    mask_path_list.append(mask_path)

    pipe.writer.SetFileName(mask_path)
    pipe.writer.Execute(mask)
    
    check_im = pipe.checkerboard.Execute( sitk.Cast(mask, sitk.sitkFloat64),
                                          inputImage,
                                          (8,8)
                                        )
    # if (display_extra):
    #   utils.display_images(sitk.GetArrayFromImage(dilate_im),
    #                       sitk.GetArrayFromImage(masked_im),
    #                       checkerboard=sitk.GetArrayFromImage(check_im),
    #                       show=True)

    mask_fig = utils.display_images(sitk.GetArrayFromImage(mask),
                                    sitk.GetArrayFromImage(masked_im),
                                    checkerboard = sitk.GetArrayFromImage(check_im),
                                    show = False
                                   )

    fig_path = os.path.join(images_path, "fig_mask_{0}.png".format(idx))
    mask_fig.savefig(fig_path)
    plt.close(mask_fig)

  df["mask_path"] = mask_path_list
  df["mask_name"] = mask_name_list

  # add mask paths
  df.to_pickle(pickle_path)

  df.to_csv(csv_path)
 
if __name__ == "__main__":
  main()