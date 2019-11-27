#!/usr/bin/env python

import SimpleITK as sitk
#import sys
import os
import utils
import tifffile as tiff
import utils
import itk
import pandas as pd
import numpy


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

    self.rescaleUIint8 = sitk.RescaleIntensityImageFilter()
    self.rescaleUIint8.SetOutputMinimum(0.0)
    self.rescaleUIint8.SetOutputMaximum(255.0)

    self.rescale_1 = sitk.RescaleIntensityImageFilter()
    self.rescale_1.SetOutputMinimum(0.0)
    self.rescale_1.SetOutputMaximum(1.0)

    self.sigmoid = sitk.SigmoidImageFilter()
    self.sigmoid.SetOutputMinimum(0.0)
    self.sigmoid.SetOutputMaximum(1.0)
    self.sigmoid.SetBeta(0.5)
    self.sigmoid.SetAlpha(0.1)# Create an itk image from the simpleitk image via numpy array

    self.gradmag = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    self.sigma = 2.0
    self.gradmag.SetSigma(self.sigma)

    self.march = sitk.FastMarchingUpwindGradientImageFilter()

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
    self.binary_fill.SetForegroundValue(255.0)

    self.dilate = sitk.BinaryDilateImageFilter()
    self.dilate.BoundaryToForegroundOff()
    self.dilate.SetKernelType(sitk.sitkBall)
    self.dilate.SetKernelRadius(1)
    self.dilate.SetForegroundValue(0)
    self.dilate.SetBackgroundValue(1)

    self.mask_neg = sitk.MaskNegatedImageFilter()
    #self.mask_neg = sitk.MaskImageFilter()
    #self.mask_neg.SetMaskingValue(255)
    self.mask_neg.SetOutsideValue(255)

    self.checkerboard = sitk.CheckerBoardImageFilter()

    self.thresh = sitk.BinaryThresholdImageFilter()
    self.thresh.SetLowerThreshold(0.0)
    self.thresh.SetUpperThreshold(0.3)
    self.thresh.SetOutsideValue(0)
    self.thresh.SetInsideValue(1)

    self.writer = sitk.ImageFileWriter()

    self.signedmaurer = sitk.SignedMaurerDistanceMapImageFilter()
    self.signedmaurer.SquaredDistanceOff()
    self.signedmaurer.UseImageSpacingOn()
    self.signedmaurer.InsideIsPositiveOff()


def main():
  # Create Masks for the images

  case_file = "case_1.pkl"
  #n_max = 256
  n_max = 128

  # top_dir = "/Volumes/SD/caseFiles"
  top_dir = "/media/store/krs/caseFiles"
  # top_dir = "/media/sansomk/510808DF6345C808/caseFiles"
  pickle_path = os.path.join(top_dir, case_file)
  df = pd.read_pickle(pickle_path)

  mask_dir = "masks"

  mask_path = os.path.join(top_dir, mask_dir)

  mask_path_list = []
  mask_name_list = []

  pipe = PipeLine()

  for idx, pd_data in df.iterrows():
    tiff_path = pd_data["crop_paths"]
    page_data = utils.get_additional_info(pd_data)

    for page in page_data:
        if( page["size_x"] > n_max):
            continue
        break
    page_idx = page["index"]
    spacing = ( page["mmp_x"], page["mmp_y"] )
    in_image = tiff.imread(tiff_path, key=page_idx)

    print(in_image.shape)

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
    #utils.display_image(sitk.GetArrayFromImage(inputImage))

    time_step = 0.5*(spacing[0] / (2.0**3.0))
    print(time_step)
    pipe.smooth.SetTimeStep(time_step)
    smoothingOutput = pipe.smooth.Execute(inputImage)

    #utils.display_image(sitk.GetArrayFromImage(smoothingOutput))

    rescale_gray = pipe.rescale_1.Execute(smoothingOutput)
    utils.display_image(sitk.GetArrayFromImage(rescale_gray))

    sigmoidOutput = pipe.sigmoid.Execute(rescale_gray)
    utils.display_image(sitk.GetArrayFromImage(sigmoidOutput))

    #invert = sitk.InvertIntensity(rescale_gray)
    #utils.display_image(sitk.GetArrayFromImage(invert))

    pipe.march.ClearTrialPoints()
    for pt in seedPositions:
      pipe.march.AddTrialPoint(pt)
    fastMarchingOutput = pipe.march.Execute(sigmoidOutput)
    utils.display_image(sitk.GetArrayFromImage(fastMarchingOutput))
    rescale_fm = pipe.rescale_1.Execute(fastMarchingOutput)
    utils.display_image(sitk.GetArrayFromImage(rescale_fm))

    #dist = pipe.signedmaurer.Execute(sitk.Cast(rescale_fm, sitk.sitkUInt32))
    #utils.display_image(sitk.GetArrayFromImage(dist))
    output = pipe.lsFilter.Execute(rescale_fm, rescale_gray)

    #output = pipe.lsFilter.Execute(sitk.InvertIntensity(fastMarchingOutput), rescale_gray)
    utils.display_image(sitk.GetArrayFromImage(output))
   
    binary_out = pipe.binary_fill.Execute(sitk.Cast(output, sitk.sitkUInt8))
    utils.display_image(sitk.GetArrayFromImage(binary_out))

    dilate_im = pipe.dilate.Execute(binary_out)
    utils.display_image(sitk.GetArrayFromImage(dilate_im))

    rescaleOutput = pipe.rescaleUIint8.Execute(dilate_im)

    masked_im = pipe.mask_neg.Execute(inputImage,
                                    sitk.Cast(dilate_im, sitk.sitkFloat64))

    mask_name = case_file.split(".")[0] + "_mask {0}.nrrd".format(idx)
    mask_path = os.path.join(mask_path, mask_name)

    mask_name_list.append(mask_path)
    mask_path_list.append(mask_path)

    pipe.writer.SetFileName(mask_path)
    pipe.writer.Execute(masked_im)
    
    check_im = pipe.checkerboard.Execute( sitk.Cast(rescaleOutput, sitk.sitkFloat64),
                                    rescale_gray,
                                    (8,8)
                                    )

    utils.display_images(sitk.GetArrayFromImage(dilate_im),
                        sitk.GetArrayFromImage(masked_im),
                        checkerboard=sitk.GetArrayFromImage(check_im))
    quit()


  df["mask_path"] = mask_path_list
  df["mask_name"] = mask_name_list

  # add mask paths
  df.to_pickle(pickle_path)
 

if __name__ == "__main__":
  main()