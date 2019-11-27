import SimpleITK as sitk
import os
import utils
import tifffile as tiff
import utils
import itk
import numpy as np

# if len(sys.argv) < 10:
#     print("Usage: {0} <inputImage> <outputImage> <seedX> <seedY> <Sigma> <SigmoidAlpha> <SigmoidBeta> <TimeThreshold>".format(sys.argv[0]))
#     sys.exit(1)

top_dir = "/media/store/krs/caseFiles/vwi_proc/"
#top_dir = "/media/sansomk/510808DF6345C808/caseFiles/vwi_proc/"

# this works for rgb, but not
fixed = os.path.join(top_dir, "case01/color_case_1_im_0000.tiff")

im_f = tiff.imread(fixed, key=8)
spacing = ( 2.02, 2.02)

#f_sitk = utils.get_sitk_image(im_f[:,:,:3], spacing = spacing, vector=True)

shape = im_f.shape
seedValue = 0
#trialPoint = (seedPosition[0], seedPosition[1], seedValue)
seedPositions = []
seedPositions.append((int(0),   int(0),   seedValue))
seedPositions.append((int(0),   int(shape[1]-1), seedValue))
seedPositions.append((int(shape[0]-1), int(0),   seedValue))
seedPositions.append((int(shape[0]-1), int(shape[1]-1), seedValue))

#sigma = 1.0
#alpha = 255.0 # 3.0/4.0 * 255.0 # width of intensity
#beta = 255.0/2.0 # center of the window
timeThreshold = 0.3
stoppingTime = 1000

# Create an itk image from the simpleitk image via numpy array
itk_image = itk.GetImageFromArray(im_f, is_vector = True)
itk_image.SetOrigin((0.0,0.0))
itk_image.SetSpacing(spacing)   
new_direction = np.eye(2)
itk_image.SetDirection(itk.matrix_from_array(new_direction))

grayscale_filter = itk.RGBToLuminanceImageFilter.New(itk_image)
grayscale_filter.Update()
grayscale = grayscale_filter.GetOutput()

# Back to a simpleitk image from the itk image
new_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(grayscale), 
    isVector = grayscale.GetNumberOfComponentsPerPixel()>1)
new_sitk_image.SetOrigin(tuple(grayscale.GetOrigin()))
new_sitk_image.SetSpacing(tuple(grayscale.GetSpacing()))
new_sitk_image.SetDirection(itk.GetArrayFromMatrix(grayscale.GetDirection()).flatten())

#img_bw =  sitk.Cast(sitk.RescaleIntensity(f_sitk), sitk.sitkUInt8)
inputImage = sitk.Cast(new_sitk_image, sitk.sitkFloat64)

#utils.display_image(sitk.GetArrayFromImage(inputImage))

#invert = sitk.InvertIntensity(inputImage)

#utils.display_image(sitk.GetArrayFromImage(invert))

#print(inputImage)

smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
smoothing.SetTimeStep(0.125)
smoothing.SetNumberOfIterations(5)
smoothing.SetConductanceParameter(9.0)
smoothingOutput = smoothing.Execute(inputImage)

utils.display_image(sitk.GetArrayFromImage(smoothingOutput))

rescale_g = sitk.RescaleIntensityImageFilter()
rescale_g.SetOutputMinimum(0.0)
rescale_g.SetOutputMaximum(255.0)
rescale_gray = rescale_g.Execute(smoothingOutput)

# gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
# gradientMagnitude.SetSigma(sigma)
# gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)

#utils.display_image(sitk.GetArrayFromImage(gradientMagnitudeOutput))

# rescale = sitk.RescaleIntensityImageFilter()
# rescale.SetOutputMinimum(0.0)
# rescale.SetOutputMaximum(1.0)
# rescaleOutput = rescale.Execute(gradientMagnitudeOutput)

# utils.display_image(sitk.GetArrayFromImage(rescaleOutput))

# sigmoid = sitk.SigmoidImageFilter()
# sigmoid.SetOutputMinimum(0.0)
# sigmoid.SetOutputMaximum(1.0)
# #sigmoid.SetAlpha(alpha)
# #sigmoid.SetBeta(beta)
# sigmoid.SetAlpha(-0.08)
# sigmoid.SetBeta(0.5)
# #sigmoid.DebugOn()

# sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)
#utils.display_image(sitk.GetArrayFromImage(sigmoidOutput))

#utils.display_image(sitk.GetArrayFromImage(sitk.InvertIntensity(sigmoidOutput)))

#bounded_reciprocal = sitk.BoundedReciprocal(rescaleOutput)
#utils.display_image(sitk.GetArrayFromImage(bounded_reciprocal))


#fastMarching = sitk.FastMarchingImageFilter()
fastMarching = sitk.FastMarchingUpwindGradientImageFilter()


for pt in seedPositions:
  fastMarching.AddTrialPoint(pt)

#fastMarching.SetStoppingValue(stoppingTime)

fastMarchingOutput = fastMarching.Execute(rescale_gray)#sigmoidOutput)

#utils.display_image(sitk.GetArrayFromImage(fastMarchingOutput))


# rescale = sitk.RescaleIntensityImageFilter()
# rescale.SetOutputMinimum(0.0)
# rescale.SetOutputMaximum(255.0)
# rescaleOutput = rescale.Execute(fastMarchingOutput)

# sitk.InvertIntensity(rescaleOutput)


lsFilter = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
lsFilter.SetMaximumRMSError(0.0)
lsFilter.SetNumberOfIterations(500)
lsFilter.SetLambda1(1)
lsFilter.SetLambda2(1)
lsFilter.SetEpsilon(1.0)
lsFilter.SetCurvatureWeight(1.0)
lsFilter.SetAreaWeight(0.0)
lsFilter.SetReinitializationSmoothingWeight(0.0)
lsFilter.SetVolume(0.0)
lsFilter.SetVolumeMatchingWeight(0.0)
lsFilter.SetHeavisideStepFunction(lsFilter.AtanRegularizedHeaviside)
output = lsFilter.Execute(fastMarchingOutput, rescale_gray)

binary_filter = sitk.BinaryFillholeImageFilter()
binary_filter.FullyConnectedOff()
binary_filter.SetForegroundValue(0.0)
binary_out = binary_filter.Execute(sitk.Cast(output, sitk.sitkUInt8))

dilate = sitk.BinaryDilateImageFilter()
dilate.BoundaryToForegroundOff()
dilate.SetKernelType(sitk.sitkBall)
dilate.SetKernelRadius(1)
dilate.SetForegroundValue(0)
dilate.SetBackgroundValue(1)

dilate_im = dilate.Execute(binary_out)

rescale = sitk.RescaleIntensityImageFilter()
rescale.SetOutputMinimum(0.0)
rescale.SetOutputMaximum(255.0)
rescaleOutput = rescale.Execute(dilate_im)

mask_input = sitk.MaskNegatedImageFilter()
#mask_input.SetMaskingValue(255)
mask_input.SetOutsideValue(255)

masksed_im = mask_input.Execute(inputImage,
                                sitk.Cast(dilate_im, sitk.sitkFloat64))

# thresholder = sitk.BinaryThresholdImageFilter()
# thresholder.SetLowerThreshold(0.0)
# thresholder.SetUpperThreshold(timeThreshold)
# thresholder.SetOutsideValue(0)
# thresholder.SetInsideValue(1)

# result = thresholder.Execute(fastMarchingOutput)

# utils.display_image(sitk.GetArrayFromImage(result))
#print(fastMarchingOutput, inputImage)
checkerboard = sitk.CheckerBoardImageFilter()
check_im = checkerboard.Execute( sitk.Cast(rescaleOutput, sitk.sitkFloat64),
                                rescale_gray,
                                (8,8)
                                )

utils.display_images(sitk.GetArrayFromImage(dilate_im),
                     sitk.GetArrayFromImage(masksed_im),
                     checkerboard=sitk.GetArrayFromImage(check_im))