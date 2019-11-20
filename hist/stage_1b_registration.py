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


# n_max is the maxmimum picture size for registration
def stage_1b_transform(reg_dict, n_max, initial_transform, count=0):
    #print(fixed[1]["crop_paths"])
    ff_path = reg_dict["f_row"]["crop_paths"]
    tf_path = reg_dict["t_row"]["crop_paths"]

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
    f_sitk = utils.get_sitk_image(im_f, spacing)

    im_t = tiff.imread(tf_path, key=page_idx)
    t_sitk = utils.get_sitk_image(im_t, spacing)

    reg_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
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
    reg_method.SetShrinkFactorsPerLevel(shrinkFactors =     [16, 16, 8, 8, 4, 4, 2, 2, 1, 1])
    # reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [3.0, 2.72727273, 2.45454545, 2.18181818, 
    #                                                         1.90909091, 1.63636364, 1.36363636, 
    #                                                         1.09090909, 0.81818182, 0.54545455, 
    #                                                         0.27272727, 0.0])
    # reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [1.96      , 1.78181818, 1.60363636, 1.42545455, 1.24727273,
    #                                                         1.06909091, 0.89090909, 0.71272727, 0.53454545, 0.35636364,
    #                                                         0.17818182, 0.        ])
    #reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2., 1., 0., 2., 1., 0., 2., 1., 0., 2., 1., 0.])
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.])
    reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Connect all of the observers so that we can perform plotting during registration.
    reg_method.AddCommand(sitk.sitkStartEvent, utils.start_plot)
    reg_method.AddCommand(sitk.sitkEndEvent, lambda: utils.end_plot(0.0))
    reg_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, utils.update_multires_iterations) 
    reg_method.AddCommand(sitk.sitkIterationEvent, lambda: utils.plot_values(reg_method))
    reg_method.SetInitialTransform(initial_transform["transform"], inPlace=False)

    min_metric = 9999999.0
    exception = True
    count = 0 
    while ((exception == True) and (count < 20)):
        try:
            # Don't optimize in-place, we would possibly like to run this cell multiple times.
            final_transform = reg_method.Execute(sitk.Cast(f_sitk, sitk.sitkFloat32), 
                                                        sitk.Cast(t_sitk, sitk.sitkFloat32))
        except RuntimeError as e:
            count += 1
            print('Got an exception\n' + str(e))
            continue
        exception = False

    measure = reg_method.GetMetricValue()
    if ( measure < min_metric):
        min_metric = measure
        best_reg = dict(
                        transform = final_transform,
                        measure = measure,
                        stop_cond = reg_method.GetOptimizerStopConditionDescription(),
                        tiff_page = page # this contains the page from the tiff file
                        )

    #2,(reg_dict["f_page"][0]["scale_x"], reg_dict["f_page"][0]["scale_y"]))
    print('Final metric value: {0}'.format(best_reg["measure"]))
    print('Optimizer\'s stopping condition, {0}'.format(best_reg["stop_cond"]))

    moving_resampled = sitk.Resample(t_sitk, f_sitk,
                                     best_reg["transform"], sitk.sitkLinear,
                                     0.0, t_sitk.GetPixelID())

    utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
                   moving_npa = sitk.GetArrayViewFromImage(moving_resampled))


    
    #print(stuff)
    #stuff = {}
    #return stuff
    while (best_reg["measure"] > -0.45):
        # try one more time
        if(count <= 0):
            best_reg = stage_1_transform(reg_dict, n_max, 1)
        else:
            #basically this isn't goog enough so try at higher resolution
            new_max = n_max*2.0
            if (new_max <= reg_dict["f_page"][0]['size_x'] ):
                print("Increasing the image resolution to {0}".format(new_max))
                best_reg = stage_1_transform(reg_dict, new_max, 0)
            else:
                print("I have run out of resolution")
                break

    return best_reg

