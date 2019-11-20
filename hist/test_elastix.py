import SimpleITK as sitk
import tifffile as tiff
import utils
import numpy as np

fixed = "/media/store/krs/caseFiles/vwi_proc/case01/color_case_1_im_0000.tiff"

moving = "/media/store/krs/caseFiles/vwi_proc/case01/color_case_1_im_0001.tiff"

im_f = tiff.imread(fixed, key=6)
im_t = tiff.imread(moving, key=6)
print(im_t.shape, im_f[:,:,:3].shape)
#quit()
spacing = ( 2.02, 2.02)
f_sitk = utils.get_sitk_image(im_f[:,:,:3], spacing = spacing, vector=True)

t_sitk = utils.get_sitk_image(im_t[:,:,:3], spacing = spacing, vector=True)


# print(t_sitk)

# num_attr = im_f.shape[2]
# fixed = sitk.Image(im_f.shape[0:2], sitk.sitkVectorUInt8, num_attr)
# moving = sitk.Image(im_t.shape[0:2], sitk.sitkVectorUInt8, num_attr)

numberOfChannels = 3

elastix = sitk.ElastixImageFilter()
for idx in range(numberOfChannels):
  elastix.AddFixedImage( sitk.VectorIndexSelectionCast(f_sitk, idx))
  elastix.AddMovingImage( sitk.VectorIndexSelectionCast(t_sitk, idx))


print('number of images:')
print( elastix.GetNumberOfFixedImages())
print( elastix.GetNumberOfMovingImages())

# read parameter file from disk so we are using the same file as command line
rigid = sitk.GetDefaultParameterMap("rigid")
rigid["DefaultPixelValue"] = ['255']
rigid["WriteResultImage"] = ['false']
affine = sitk.GetDefaultParameterMap("affine")
affine["NumberOfResolutions"] = ['5']
affine["DefaultPixelValue"] = ['255']
affine["WriteResultImage"] = ['false']
sitk.PrintParameterMap(affine)
#elastix.SetParameterMap(rigid)
#quit()
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(rigid)
parameterMapVector.append(affine)

elastix.SetParameterMap(parameterMapVector)

#elastix.SetParameterMap(elastix.ReadParameterFile('params_6.txt'))

elastix.SetOutputDirectory('./result/')
elastix.LogToConsoleOff()
elastix.SetLogToFile(True)
elastix.Execute()

moving_resampled = elastix.GetResultImage()
tran_map = elastix.GetTransformParameterMap()
test_image = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 0), tran_map)
test_image1 = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 1), tran_map)
test_image2 = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 2), tran_map)

new_array = np.empty_like(im_f[:,:,:3])
new_array[:,:,0] = sitk.GetArrayViewFromImage(test_image)
new_array[:,:,1] = sitk.GetArrayViewFromImage(test_image1)
new_array[:,:,2] = sitk.GetArrayViewFromImage(test_image2)
utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
               moving_npa = new_array)