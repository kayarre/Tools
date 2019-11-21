import pickle 
#import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
#import itk

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

import utils #import get_sitk_image, display_images

elastic_dir = "/media/sansomk/510808DF6345C808/caseFiles/elastix"


# n_max is the maxmimum picture size for registration
def stage_1b_transform(reg_dict, n_max, initial_transform, count=0):
    #print(fixed[1]["crop_paths"])
    # load color images
    ff_path = reg_dict["f_row"]["color_paths"]
    tf_path = reg_dict["t_row"]["color_paths"]

    #base_res_x = reg_dict["f_row"]["mpp-x"]
    #base_res_y = reg_dict["f_row"]["mpp-y"]

    for page in reg_dict["f_page"]:
        if( page["size_x"] > n_max):
            continue
        break
    page_idx = page["index"]

    print(page_idx, page)
    spacing = ( page["mmp_x"], page["mmp_y"] )
    print(spacing)
    # transform numpy array to simpleITK image
    # have set the parameters manually
    im_f = tiff.imread(ff_path, key=page_idx)
    #f_sitk = utils.get_sitk_image(im_f, spacing)
    f_sitk = utils.get_sitk_image(im_f[:,:,:3], spacing = spacing, vector=True)

    im_t = tiff.imread(tf_path, key=page_idx)
    #t_sitk = utils.get_sitk_image(im_t, spacing)
    t_sitk = utils.get_sitk_image(im_t[:,:,:3], spacing = spacing, vector=True)

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
    #sitk.PrintParameterMap(rigid)
    
    transform_init = initial_transform["transform"]
    in_params = list(transform_init.GetParameters())
    fixed_in_params = transform_init.GetFixedParameters()
    rigid["TransformParameters"]  = [ str(a) for a in in_params]
    rigid["CenterOfRotationPoint"]  = [ str(a) for a in fixed_in_params]


    rigid_in_params = {}
    for key, value in rigid.items():
      rigid_in_params[key] = value
      #print(rigid["TransformParameters"])

    # affine = sitk.GetDefaultParameterMap("affine")
    # affine["NumberOfResolutions"] = ['5']
    # affine["DefaultPixelValue"] = ['255']
    # affine["WriteResultImage"] = ['false']

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(rigid)
    # parameterMapVector.append(affine)

    elastix.SetParameterMap(parameterMapVector)

    elastix.SetOutputDirectory(elastic_dir)
    elastix.LogToConsoleOff()
    elastix.SetLogToFile(True)

    min_metric = 9999999.0
    exception = True
    count = 0 
    while ((exception == True) and (count < 20)):
        try:
            # Don't optimize in-place, we would possibly like to run this cell multiple times.
            elastix.Execute()
        except RuntimeError as e:
            count += 1
            print('Got an exception\n' + str(e))
            continue
        exception = False

    #moving_resampled = elastix.GetResultImage()
    tran_map = elastix.GetTransformParameterMap()
    print(tran_map)
    print("\n")
    print(sitk.PrintParameterMap(tran_map))
    trans_out_params = {}
    print(len(tran_map))
    for idx, t in enumerate(tran_map):
        trans = {}
        for key, value in t.items():
            trans[key] = value
        trans_out_params[idx] = trans

    print(rigid_in_params)
    print("\n")
    print(trans_out_params)

    #print(tran_map.GetTransformParameter("CenterOfRotationPoint"))
    #print(tran_map["TransformParameters"])
    test_image = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 0), tran_map)
    test_image1 = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 1), tran_map)
    test_image2 = sitk.Transformix(sitk.VectorIndexSelectionCast(t_sitk, 2), tran_map)

    new_array = np.empty_like(im_f[:,:,:3])
    new_array[:,:,0] = sitk.GetArrayViewFromImage(test_image)
    new_array[:,:,1] = sitk.GetArrayViewFromImage(test_image1)
    new_array[:,:,2] = sitk.GetArrayViewFromImage(test_image2)
    utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
                moving_npa = new_array)


    # min_metric = 9999999.0
    # exception = True
    # count = 0 
    # while ((exception == True) and (count < 20)):
    #     try:
    #         # Don't optimize in-place, we would possibly like to run this cell multiple times.
    #         final_transform = reg_method.Execute(sitk.Cast(f_sitk, sitk.sitkFloat32), 
    #                                                     sitk.Cast(t_sitk, sitk.sitkFloat32))
    #     except RuntimeError as e:
    #         count += 1
    #         print('Got an exception\n' + str(e))
    #         continue
    #     exception = False

    # measure = reg_method.GetMetricValue()
    # if ( measure < min_metric):
    #     min_metric = measure
    #     best_reg = dict(
    #                     transform = final_transform,
    #                     measure = measure,
    #                     stop_cond = reg_method.GetOptimizerStopConditionDescription(),
    #                     tiff_page = page # this contains the page from the tiff file
    #                     )

    # #2,(reg_dict["f_page"][0]["scale_x"], reg_dict["f_page"][0]["scale_y"]))
    # print('Final metric value: {0}'.format(best_reg["measure"]))
    # print('Optimizer\'s stopping condition, {0}'.format(best_reg["stop_cond"]))

    # moving_resampled = sitk.Resample(t_sitk, f_sitk,
    #                                  best_reg["transform"], sitk.sitkLinear,
    #                                  0.0, t_sitk.GetPixelID())

    # utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
    #                moving_npa = sitk.GetArrayViewFromImage(moving_resampled))


    
    # #print(stuff)
    # #stuff = {}
    # #return stuff
    # while (best_reg["measure"] > -0.45):
    #     # try one more time
    #     if(count <= 0):
    #         best_reg = stage_1_transform(reg_dict, n_max, 1)
    #     else:
    #         #basically this isn't goog enough so try at higher resolution
    #         new_max = n_max*2.0
    #         if (new_max <= reg_dict["f_page"][0]['size_x'] ):
    #             print("Increasing the image resolution to {0}".format(new_max))
    #             best_reg = stage_1_transform(reg_dict, new_max, 0)
    #         else:
    #             print("I have run out of resolution")
    #             break

    return trans_out_params

