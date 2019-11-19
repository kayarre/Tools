import pickle 
#import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import itk

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

import utils #import get_sitk_image, display_images, start_plo


# n_max is the maxmimum picture size for registration
def stage_2_transform(reg_dict, n_max, initial_transform):
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
    #f_cog = get_center_of_gravity(im_f, spacing)
    f_sitk = utils.get_sitk_image(im_f, spacing)

    im_t = tiff.imread(tf_path, key=page_idx)
    t_sitk = utils.get_sitk_image(im_t, spacing)
    #bin_est = np.sqrt(np.prod(t_sitk.GetSize())/5.0)
    bin_est = 50

    reg_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bin_est)
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
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.])
    reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Connect all of the observers so that we can perform plotting during registration.
    reg_method.AddCommand(sitk.sitkStartEvent, utils.start_plot)
    reg_method.AddCommand(sitk.sitkEndEvent, lambda: utils.end_plot(angle))
    reg_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, utils.update_multires_iterations) 
    reg_method.AddCommand(sitk.sitkIterationEvent, lambda: utils.plot_values(reg_method))

    min_metric = 9999999.0
    for angle in n_angles:
        rot.SetAngle(angle)
        #print(initial_transform)
        try:
            # Don't optimize in-place, we would possibly like to run this cell multiple times.
            reg_method.SetInitialTransform(rot, inPlace=False)
            final_transform = reg_method.Execute(sitk.Cast(f_sitk, sitk.sitkFloat32), 
                                                        sitk.Cast(t_sitk, sitk.sitkFloat32))
        except RuntimeError as e:
            print('Got an exception\n' + str(e))
            continue
        measure = reg_method.GetMetricValue()
        if ( measure < min_metric):
            min_metric = measure
            best_reg = dict( angle = angle,
                            transform = final_transform,
                            measure = measure,
                            stop_cond = reg_method.GetOptimizerStopConditionDescription()
                            )
        print(measure, reg_method.GetOptimizerStopConditionDescription())
        # t_resampled = sitk.Resample(t_sitk, f_sitk,
        #                              final_transform, sitk.sitkLinear,
        #                              0.0, t_sitk.GetPixelID())
        # display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
        #            moving_npa = sitk.GetArrayViewFromImage(t_resampled))

        # f_rescale = sitk.RescaleIntensityImageFilter()
        # f_rescale.SetOutputMaximum(1)
        # f_rescale.SetOutputMinimum(0)

    print(best_reg["transform"])
    print('Final metric value: {0}'.format(best_reg["measure"]))
    print('Optimizer\'s stopping condition, {0}'.format(best_reg["stop_cond"]))

    moving_resampled = sitk.Resample(t_sitk, f_sitk,
                                     best_reg["transform"], sitk.sitkLinear,
                                     0.0, t_sitk.GetPixelID())

    # display_images_with_alpha(alpha = (0.0, 1.0, 0.05),
    #                           fixed = f_sitk, moving = t_sitk)

    display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
                   moving_npa = sitk.GetArrayViewFromImage(moving_resampled))

    # im_test = resample(t_sitk, rot)
    # myshow(im_test)
    print("test")

    
    #print(stuff)
    #stuff = {}
    #return stuff
    if (best_reg["measure"] > -0.50):
        #basically this isn't goog enough so try at higher resolution
        # increase resolution
        new_max = n_max*2.0
        if (new_max <= reg_dict["f_page"][0]['size_x'] ):
            print("increasing the image resolution to {0}".format(new_max))
            best_reg = stage_1_transform(reg_dict, new_max)
        else:
            print("I have run out of resolution")
            return best_reg
    else:
        return best_reg

