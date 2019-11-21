import pickle 
#import pyvips
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
from multiprocessing.pool import ThreadPool
from functools import partial

import logging
logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.DEBUG)

import utils #import get_sitk_image, display_images

# This function evaluates the metric value in a thread safe manner
def evaluate_metric(current_rotation, tx, f_image, m_image):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)
    registration_method.SetInterpolator(sitk.sitkLinear)
    current_transform = sitk.Euler2DTransform(tx)
    current_transform.SetAngle(current_rotation)
    registration_method.SetInitialTransform(current_transform)
    res = registration_method.MetricEvaluate(f_image, m_image)
    return res

# Threads of threads ?????
# don't know if this makes sense
def evaluate_metric_extra(current_page, f_image, m_image, angles):
    page_idx = current_page["index"]
    spacing = ( current_page["mmp_x"], current_page["mmp_y"] )
    # transform numpy array to simpleITK image
    # have set the parameters manually
    im_f = tiff.imread(f_image, key=page_idx)
    f_sitk = utils.get_sitk_image(im_f, spacing)

    im_t = tiff.imread(m_image, key=page_idx)
    t_sitk = utils.get_sitk_image(im_t, spacing)

    initial_transform = sitk.CenteredTransformInitializer(f_sitk, 
                                                    t_sitk, 
                                                    sitk.Euler2DTransform(), 
                                                    sitk.CenteredTransformInitializerFilter.MOMENTS)

    p = ThreadPool(len(angles))
    all_metric_values = p.map(partial(evaluate_metric, 
                                    tx = initial_transform, 
                                    f_image = sitk.Cast(f_sitk, sitk.sitkFloat32),
                                    m_image = sitk.Cast(t_sitk, sitk.sitkFloat32)),
                            angles)
    #print(all_metric_values)
    best_orientation = angles[np.argmin(all_metric_values)]
    #print('best orientation is: ' + str(best_orientation))
    param_test = {}
    param_test[page_idx] = dict(angle=best_orientation,
                                metric = np.min(all_metric_values),
                                size_x=current_page["size_x"],
                                size_y=current_page["size_y"])

    return param_test

# n_max is the maxmimum picture size for registration
def stage_1_parallel_metric(reg_dict, n_max, count=0):
    #print(fixed[1]["crop_paths"])
    ff_path = reg_dict["f_row"]["crop_paths"]
    tf_path = reg_dict["t_row"]["crop_paths"]

    #base_res_x = reg_dict["f_row"]["mpp-x"]
    #base_res_y = reg_dict["f_row"]["mpp-y"]
    param_test = {}
    # get the best one over 30 rotationss
    n_angles = list(np.linspace(0.0, 2.0*np.pi * 63.0/64.0, 64))
    page_list = []
    for page in reg_dict["f_page"]:
        if( page["size_x"] > n_max):
            continue
        page_list.append(page)


    p_page = ThreadPool(len(page_list))

    list_dicts = p_page.map(partial(evaluate_metric_extra, 
                                    f_image = ff_path,
                                    m_image = tf_path,
                                    angles = n_angles),
                            page_list)

    result = {}
    metric = 99999999.0
    keep = None
    idx = None
    for d in list_dicts:
        result.update(d)
        for k, data in d.items():
            if (data["metric"] < metric):
                metric = data["metric"]
                keep = d[k]
                idx = k
    result["n_angles"] = len(n_angles)
    result["best_angle"] = keep["angle"]
    result["best_metric"] = metric
    result["best_page_idx"] = idx

    return result
