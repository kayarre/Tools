import pickle 
#import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk

# from ipywidgets import interact, fixed
# from IPython.display import clear_output

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

import utils #import get_sitk_image, display_images, start_plo


# n_max is the maxmimum picture size for registration
def stage_3_transform(reg_dict, n_max, initial_transform, count=0):
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

    t_trans = utils.resample(t_sitk, initial_transform["transform"],
                              default_value = 0.0,
                              interpolator = sitk.sitkLanczosWindowedSinc,
                              ref_image = f_sitk)

    # does the copy constructor work?
    transformCopy = initial_transform["transform"]

    spline_order = 1
    bin_est = 50
    trans_domain_mesh_sz = [10]*t_sitk.GetDimension()
    bspline = sitk.BSplineTransformInitializer(f_sitk,
                                              transformDomainMeshSize = trans_domain_mesh_sz,
                                              order = spline_order)

    reg_method = sitk.ImageRegistrationMethod()
    reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bin_est)
    reg_method.SetMetricSamplingStrategy(reg_method.RANDOM) #REGULAR
    reg_method.SetMetricSamplingPercentage(0.2)
    reg_method.SetInterpolator(sitk.sitkLinear)
    reg_method.SetOptimizerAsGradientDescent(learningRate=5.0,
                                            numberOfIterations=100,
                                            convergenceMinimumValue=1e-10,
                                            convergenceWindowSize=20)
    reg_method.SetOptimizerScalesFromPhysicalShift()
    reg_method.SetInterpolator(sitk.sitkBSpline)
    reg_method.SetShrinkFactorsPerLevel(shrinkFactors = [10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 1, 1])
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Connect all of the observers so that we can perform plotting during registration.
    reg_method.AddCommand(sitk.sitkStartEvent, utils.start_plot)
    reg_method.AddCommand(sitk.sitkEndEvent, lambda: utils.end_plot(0.0))
    reg_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, utils.update_multires_iterations) 
    reg_method.AddCommand(sitk.sitkIterationEvent, lambda: utils.plot_values(reg_method))

    reg_method.SetInitialTransform(bspline)

    min_metric = 9999999.0
    exception = True
    count = 0 
    while ((exception == True) and (count < 20)):
        try:
            # Don't optimize in-place, we would possibly like to run this cell multiple times.
            final_transform = reg_method.Execute(sitk.Cast(f_sitk, sitk.sitkFloat64), 
                                                        sitk.Cast(t_trans, sitk.sitkFloat64))
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
    print(measure, reg_method.GetOptimizerStopConditionDescription())
    t_resampled = sitk.Resample(t_sitk, f_sitk,
                                 final_transform, sitk.sitkLinear,
                                 0.0, t_sitk.GetPixelID())
    utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
               moving_npa = sitk.GetArrayViewFromImage(t_resampled))
    
    composite = sitk.Transform(initial_transform["transform"].GetDimension(), sitk.sitkComposite)
    composite.AddTransform(initial_transform["transform"])
    composite.AddTransform(final_transform)

    final_transform.AddTransform(initial_transform["transform"])
    t_resampled2 = sitk.Resample(t_sitk, f_sitk,
                                 composite, sitk.sitkLinear,
                                 0.0, t_sitk.GetPixelID())
    utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
               moving_npa = sitk.GetArrayViewFromImage(t_resampled2))

    print(best_reg["transform"])
    print('Final metric value: {0}'.format(best_reg["measure"]))
    print('Optimizer\'s stopping condition, {0}'.format(best_reg["stop_cond"]))

    moving_resampled = sitk.Resample(t_sitk, f_sitk,
                                     best_reg["transform"], sitk.sitkLinear,
                                     0.0, t_sitk.GetPixelID())

    # display_images_with_alpha(alpha = (0.0, 1.0, 0.05),
    #                           fixed = f_sitk, moving = t_sitk)

    utils.display_images(fixed_npa = sitk.GetArrayViewFromImage(f_sitk),
                   moving_npa = sitk.GetArrayViewFromImage(moving_resampled))

    # im_test = resample(t_sitk, rot)
    # myshow(im_test)
    print("test")

    
    #print(stuff)
    #stuff = {}
    #return stuff
    while (best_reg["measure"] > -0.45):
        # try one more time
        if(count <= 0):
            best_reg = stage_3_transform(reg_dict, n_max, initial_transform, 1)
        else:
            #basically this isn't goog enough so try at higher resolution
            new_max = n_max*2.0
            if (new_max <= reg_dict["f_page"][0]['size_x'] ):
                print("Increasing the image resolution to {0}".format(new_max))
                best_reg = stage_3_transform(reg_dict, new_max, initial_transform, 0)
            else:
                print("I have run out of resolution")
                break

    return best_reg


def demons_registration(fixed_image, moving_image, fixed_points = None, moving_points = None):
    
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))
    
    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0) 
    
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU
        
    # Multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])    

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)        
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
        
    return registration_method.Execute(fixed_image, moving_image)

def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    """
    Args:
        image: The image we want to resample.
        shrink_factor(s): Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma(s): Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
    """
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors]*image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
    
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(shrink_factors,original_size)]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0, 
                         image.GetPixelID())

def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None, 
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors (list of lists or scalars): Shrink factors relative to the original image's size. When the list entry, 
                                                   shrink_factors[i], is a scalar the same factor is applied to all axes.
                                                   When the list entry is a list, shrink_factors[i][j] is applied to axis j.
                                                   This allows us to specify different shrink factors per axis. This is useful
                                                   in the context of microscopy images where it is not uncommon to have
                                                   unbalanced sampling such as a 512x512x8 image. In this case we would only want to 
                                                   sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
        smoothing_sigmas (list of lists or scalars): Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns: 
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))
    
    # Create initial displacement field at lowest resolution. 
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(), 
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])
 
    # Run the registration.            
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1], 
                                                                moving_images[-1], 
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.    
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
            initial_displacement_field = sitk.Resample (initial_displacement_field, f_image)
            initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)