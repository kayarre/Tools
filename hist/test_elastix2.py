import SimpleITK as sitk
import tifffile as tiff
import utils

fixed = "/media/store/krs/caseFiles/vwi_proc/case01/case_1_im_0000.tiff"

moving = "/media/store/krs/caseFiles/vwi_proc/case01/case_1_im_0001.tiff"

im_f = tiff.imread(fixed, key=4)
im_t = tiff.imread(moving, key=4)
# print(im_t.shape, im_f[:,:,:3].shape)
#quit()
spacing = ( 2.02, 2.02)
f_sitk = utils.get_sitk_image(im_f, spacing = spacing)

t_sitk = utils.get_sitk_image(im_t, spacing = spacing)

# print(t_sitk)

#numberOfChannels = 3

elastix = sitk.ElastixImageFilter()
elastix.SetFixedImage(f_sitk)
elastix.SetMovingImage(t_sitk)

print('number of images:')
print( elastix.GetNumberOfFixedImages())
print( elastix.GetNumberOfMovingImages())

# read parameter file from disk so we are using the same file as command line
#elastix.SetParameterMap(elastix.ReadParameterFile('params_6.txt'))
rigid = sitk.GetDefaultParameterMap("rigid")
affine = sitk.GetDefaultParameterMap("affine")
sitk.PrintParameterMap(rigid)
#quit()
#sitk.PrintParameterMap(affine)
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(rigid)
#parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
#parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastix.SetParameterMap(parameterMapVector)

elastix.SetOutputDirectory('./result/')
elastix.LogToConsoleOff()
elastix.SetLogToFile(True)

elastix.Execute()

sitk.PrintParameterMap(elastix.GetTransformParameterMap())

moving_resampled = elastix.GetResultImage()

utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
               moving_npa = sitk.GetArrayViewFromImage(moving_resampled))