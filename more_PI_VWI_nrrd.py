import nrrd
import json
import numpy as np
import pickle
import os

import scipy.stats as stats
from scipy.optimize import fsolve
from scipy.special import iv
from scipy.special import factorial2, factorial
from scipy.special import hyp1f1
from sklearn.neighbors.kde import KernelDensity
#from sklearn.model_selection import GridSearchCV


#from dask.distributed import Client
#client = Client('128.95.156.220:8785')

#import joblib

import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd


def VWI_Enhancement(post, pre, mean_post_vent, mean_pre_vent, kind = "E1",
                                      std_post_vent = None, std_pre_vent = None, return_parts = False):
    """ calculate enhancement 
    Parameters
    ----------
    post : numpy array_like post contrast VWI
    pre : numpy array_like pre contrast VWI
    mean_post_vent : mean of post contrast ventricle
    mean_pre_vent : mean of pre contrast ventricle
    kind : which enhancement to calculate
    std_post_vent : mean of post contrast ventricle
    std_pre_vent : mean of pre contrast ventricle
    -------
    returns the enhancement calculation, numpy array_like
    """
    if kind == "E1":
        #"E = xi_vent / eta_vent * eta - xi"
        post_ =  (mean_pre_vent / mean_post_vent * post)
        pre_ = pre
    elif kind == "E2":
        #"E = eta / eta_vent - xi / xi_vent"
        post_ = (post / mean_post_vent)
        pre_ = (pre / mean_pre_vent)
    elif kind == "E3":
        #"E = ( eta - mean_eta_vent) / stddev(eta_vent) - (xi - mean_xi_vent) / stddev(xi_vent)"
        post_ = (post - mean_post_vent) / std_post_vent
        pre_ = (pre - mean_pre_vent) / std_pre_vent 
    elif kind == "E5":
        # ratio of normalized things, similar to E3
        E = ( std_pre_vent / std_post_vent ) * (post - mean_post_vent) / (pre - mean_pre_vent)
        return E
    elif kind == "E4":
        # ratio of normalized things, similar to E3
        num = np.sqrt(std_post_vent**2 + std_pre_vent**2)
        post_ = (post - mean_post_vent) / num
        pre_ = (pre - mean_pre_vent) / num 
    else:
        raise  Exception("undefined enhancement kind {0}".format(kind))
    
    E = post_ - pre_
    if return_parts:
        return E, post_, pre_
    else:
        return E

def div0( a, b, value=0 ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = value  # -inf inf NaN
    return c


def kernel_fit_single(data, bw=None, min_size=20, kern='gaussian'):
    """ guassian fit to 1D data
    """
    res = np.histogram(data.ravel(), bins='sqrt', density=True)
    std_data = data.std()
    if (bw == None):
        bw = (data.ravel().shape[0] * (std_data+ 2) / 4.)**(-1. / (std_data + 4))

    N_bins = res[1].shape[0]
    if (N_bins < min_size):
        extra = 0.2
        #N_bins *=2
    else:
        extra = 0.0
    # get plus or minus 20%

    x_grid = np.linspace(res[1][0]-extra*abs(res[1][0]), res[1][-1] + extra*abs(res[1][0]), N_bins)

    kde = KernelDensity(bandwidth=bw, kernel=kern)
    kde.fit(data.ravel()[:, None])

    pdf = np.exp(kde.score_samples(x_grid[:, None]))

    return pdf, x_grid

def ape(actual, predicted):
    """
    Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
    """
    return np.arctan((actual - predicted) / (actual + np.finfo(float).eps))

def aape(actual, predicted):
    """
    Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
    """
    return np.arctan(np.abs(actual - predicted) / (actual + np.finfo(float).eps))

def add_grade(case_id, E, wall, wall_mean, mean_PI, grades, count_old=None, n_grade=3):
    model_class = np.zeros_like(wall, dtype=np.uint8)
    model_class[ (wall > wall_mean) &
                 (wall < mean_PI) ] = np.uint8(1)
    model_class[ (wall >=  mean_PI) ] = np.uint8(2)
    bin_cnt = np.bincount(model_class, minlength=n_grade)
    bin_cnt_sum = bin_cnt.sum()
 
    count_list = []
    min_float = 1.0 / np.finfo(float).max
    eps = 0.0
    print(count_old)
    for grade, count in enumerate(bin_cnt):
        percent_new = float(count) / float(bin_cnt_sum)
        if ( count_old == None):
            percent_change = 0.0
            count_list.append(float(count))
            aape_res = aape( count_list[-1], count_list[-1])
            ape_res = ape(count_list[-1], count_list[-1])
        else:
            if ( count_old[grade] > 0.0):
               eps = 0.0
            else:
               eps = min_float

            #percent_change =  ( (percent_new - percent_old[grade]) / (np.abs(percent_old[grade]) + eps)) * 100.0
            percent_change =  ( (count - count_old[grade]) / (count_old[grade]+ eps)) * 100.0
            aape_res = aape( count_old[grade], count)
            ape_res = ape( count_old[grade], count)
        print([case_id, E, grade, count, percent_new*100.0, percent_change, aape_res, ape_res])
        grades.append([case_id, E, grade, count, percent_new*100.0, percent_change, aape_res, ape_res ])
    if (count_old == None):
        return count_list

#vars

conf = 2.0 #95% confidence interval
percent_boot = 0.01
font_size = 10

figure_1 = True
figure_2 = True

dpi_value = 200
json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_normalization.json"

pickle_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_pickle.pkl"

enhancement_file = "enhancement_pickle.pkl"
bw_file = "bw_pickle.pkl"
write_file_dir = "/home/sansomk/caseFiles/mri/VWI_proj"
write_dir = "VWI_analysis"
plots_dir = "plots"
overwrite = 0
overwrite_out = True
skip_bootstrap = True

skip_write = True
process_grade_file = os.path.join(write_file_dir, "clinical_grade.pkl")
process_grade_file_list = os.path.join(write_file_dir, "clinical_grade_list.pkl")

with open(json_file, 'r') as f:
    data = json.load(f)

with open(os.path.join(write_file_dir, bw_file), 'rb') as handle:
    params_list = pickle.load( handle)
print("load bandwidght parameters")    

labels = ["post_float","pre_float", "VWI_post_masked_vent", "VWI_post_vent",
        "pre2post", "level_set", "VWI_pre2post_masked_vent",
        "VWI_background_post_masked", "VWI_background_post",
        "VWI_background_pre_masked", "VWI_background_pre", 
        "model-label_cropped", "model_post_masked",
        "model_pre2post_masked",  "VWI_post_float_cropped", "VWI_background_intersection",
        "VWI_background_post_intersection", "VWI_background_pre_intersection"]

subset_labels = ["VWI_post_masked_vent", "VWI_pre2post_masked_vent",
    "VWI_background_post_masked", "VWI_background_pre_masked",
    "model_post_masked", "model_pre2post_masked"]

groups = [("post_float", "pre2post"), 
                ("VWI_post_masked_vent", "VWI_pre2post_masked_vent"),
                ("model_post_masked", "model_pre2post_masked"),
                ("VWI_background_post_masked", "VWI_background_pre_masked"), 
                ("VWI_background_post_intersection", "VWI_background_pre_intersection"),
                ("VWI_post_PI_masked", "VWI_pre2post_PI_masked")
                ]
image_dict = {}



if ((not os.path.exists(pickle_file)) or overwrite == 1):
    for case_id, images in data.items():
        image_dict[case_id] = {}
        print(case_id)
        for post_label, pre_label  in groups:
            #print(pre_label, post_label)
            pre_path = images[pre_label]
            post_path = images[post_label]    
            #vwi_mask, vwi_mask_header = nrrd.read(vwi_mask_path)
            image_tuple_pre = nrrd.read(pre_path)
            image_tuple_post = nrrd.read(post_path)
            
            image_dict[case_id][pre_label] = image_tuple_pre
            image_dict[case_id][post_label] = image_tuple_post

    pickle.dump(image_dict,  open(pickle_file, "wb"))

else:
    with open(pickle_file, "rb") as pkl_f:
        image_dict = pickle.load(pkl_f)
        pickle_time = os.path.getmtime(pickle_file)
      
    dump = False
    for case_id, images in data.items():
        print(case_id)
        for post_label, pre_label  in groups:
            #print(pre_label, post_label)numpy ones like
            pre_path = images[pre_label]
            pre_time = os.path.getmtime(pre_path)
            post_path = images[post_label]
            post_time = os.path.getmtime(post_path)
            #vwi_mask, vwi_mask_header = nrrd.read(vwi_mask_path)
            if ( pre_label not in image_dict[case_id].keys() or pre_time > pickle_time):
                image_tuple_pre = nrrd.read(pre_path)
                image_dict[case_id][pre_label] = image_tuple_pre
                dump = True
                
            if (post_label not in image_dict[case_id].keys()  or post_time > pickle_time):
                image_tuple_post = nrrd.read(post_path)
                image_dict[case_id][post_label] = image_tuple_post
                dump = True

    if (dump ):
        pickle.dump(image_dict,  open(pickle_file, "wb"))
      
    


channels = int(1)
pi_columns = ["Case ID", "Enhancement Type" , "Grade", "Count", "Percent",
              "Percent Change", "Arctangent Absolute Percentage Difference",
              "Arctangent Percentage Difference"]
image_path_list = []
grade_list = []
eps = np.finfo(float).eps

for case_id, case_imgs  in image_dict.items():
    #full_image = np.zeros_like(case_imgs["post_float"][0])
    
    img_post_vent = case_imgs["VWI_post_masked_vent"][0][case_imgs["VWI_post_masked_vent"][0] >= 0.0]
    img_post_back = case_imgs["VWI_background_post_masked"][0][case_imgs["VWI_background_post_masked"][0] >= 0.0]
    img_post_model = case_imgs["model_post_masked"][0][case_imgs["model_post_masked"][0] >= 0.0]
    img_post_back_inter = case_imgs["VWI_background_post_intersection"][0][case_imgs["VWI_background_post_intersection"][0] >= 0.0]
    
    img_pre_vent = case_imgs["VWI_pre2post_masked_vent"][0][case_imgs["VWI_pre2post_masked_vent"][0] >= 0.0]
    img_pre_back = case_imgs["VWI_background_pre_masked"][0][case_imgs["VWI_background_pre_masked"][0] >= 0.0]
    img_pre_model = case_imgs["model_pre2post_masked"][0][case_imgs["model_pre2post_masked"][0] >= 0.0]
    img_pre_back_inter = case_imgs["VWI_background_pre_intersection"][0][case_imgs["VWI_background_pre_intersection"][0] >= 0.0]
    
    eta = case_imgs["post_float"][0][case_imgs["post_float"][0] >= 0.0]
    xi = case_imgs["pre2post"][0][case_imgs["pre2post"][0] >= 0.0]

    non_zero_indx = np.where(case_imgs["post_float"][0] >= 0.0)

    
    eta_model = case_imgs["model_post_masked"][0][case_imgs["model_post_masked"][0] >= 0.0]
    xi_model = case_imgs["model_pre2post_masked"][0][case_imgs["model_pre2post_masked"][0] >= 0.0]
    
    #non_zero_indx_model= np.where(case_imgs["model_post_masked"][0] >= 0.0)
    
    eta_PI = case_imgs["VWI_post_PI_masked"][0][case_imgs["VWI_post_PI_masked"][0] >= 0.0]
    xi_PI = case_imgs["VWI_pre2post_PI_masked"][0][case_imgs["VWI_pre2post_PI_masked"][0] >= 0.0]
    
    #scale_factor = np.sqrt(2.0-np.pi/2.0) 
    #post_back_noise = np.mean(img_post_back) / np.sqrt(np.pi/2.0) # rayleigh distribution
    #pre_back_noise = np.mean(img_pre_back) / np.sqrt(np.pi/2.0) # rayleigh distribution
    
    back_std_pre = np.std(img_pre_back) 
    back_std_post = np.std(img_post_back)
    
    #print(case_id, "post vent MEAN: {0:.4f} pre vent MEAN {1:.4f}".format(np.mean(img_post_vent), np.mean(img_pre_vent)))
    #print(case_id, "post vent STD: {0:.4f} pre vent STD {1:.4f}".format(np.std(img_post_vent) , np.std(img_pre_vent) ))
    
    #print(case_id, "post back MEAN: {0:.4f} pre back MEAN {1:.4f}".format(np.mean(img_post_back) , np.mean(img_pre_back) ))
    #print(case_id, "post inter MEAN: {0:.4f} pre inter MEAN {1:.4f}".format(np.mean(img_post_back_inter) , np.mean(img_pre_back_inter) ))
    ##print(case_id, "post vent inter shape: {0} pre vent inter shape {1}".format(img_post_back_inter.shape ,img_pre_back_inter.shape ))
    
    #print(case_id, "post back STD: {0:.4f} pre back STD {1:.4f}".format(back_std_post, back_std_pre))
    #print(case_id, "post inter STD: {0:.4f} pre inter STD {1:.4f}".format(np.std(img_post_back_inter) , np.std(img_pre_back_inter) ))
    
    #print(case_id, "post PI Mean: {0:.4f} pre PI mean {1:.4f}".format(np.mean(eta_PI), np.mean(xi_PI)))
    #print(case_id, "post PI STD: {0:.4f} pre PI STD {1:.4f}".format(np.std(eta_PI), np.std(xi_PI)))


    eta_vent = np.mean(img_post_vent) # mean ventricle post
    xi_vent = np.mean(img_pre_vent) # mean ventricle pre

    ## ventricle portion
    cov_vent = np.cov(np.vstack((img_post_vent, img_pre_vent)))
    #N_vent = float(img_post_vent.shape[0])

    #print(np.sqrt(2.0*np.trace(cov_vent) - np.sum(cov_vent)))

    eta_vent = np.mean(img_post_vent) # mean ventricle post
    xi_vent = np.mean(img_pre_vent) # mean ventricle pre
    u_eta_vent_2 = cov_vent[0,0] #/ N_vent  # uncertainty vent post square
    u_xi_vent_2 = cov_vent[1,1]  #/ N_vent  # uncertainty vent pre square
    u_eta_xi_vent = cov_vent[0,1] #/ N_vent # covariance between pre and post
    
    
    E_model, E_post_model, E_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                           1.0, 1.0, kind = "E1", return_parts=True)
    E1_model, E1_post_model, E1_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E1", return_parts=True)
    
    E2_model, E2_post_model, E2_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E2", return_parts=True)

    E3_model, E3_post_model, E3_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    
    E4_model, E4_post_model, E4_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)


    E_PI, E_post_PI, E_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           1.0, 1.0, kind = "E1", return_parts=True)

    E1_PI, E1_post_PI, E1_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           eta_vent, xi_vent, kind = "E1", return_parts=True)
    E2_PI, E2_post_PI, E2_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           eta_vent, xi_vent, kind = "E2", return_parts=True)
    E3_PI, E3_post_PI, E3_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                            eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    
    E4_PI, E4_post_PI, E4_pre_PI  = VWI_Enhancement(eta_PI, xi_PI,
                                                            eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)

    clinical_model_fit, clinical_grid = kernel_fit_single(eta_model, params_list[case_id]["E"]["model"]["bandwidth"])
    m_idx = np.argmax(clinical_model_fit)
    mean_model_clinical = (clinical_grid[m_idx] + clinical_grid[m_idx+1]) / 2.0

    E_model_fit, E_grid = kernel_fit_single(E_model, params_list[case_id]["E"]["model"]["bandwidth"])
    m_idx = np.argmax(E_model_fit)
    mean_model_E = (E_grid[m_idx] + E_grid[m_idx+1]) / 2.0  

    E1_model_fit, E1_grid = kernel_fit_single(E1_model, params_list[case_id]["E1"]["model"]["bandwidth"])
    m_idx = np.argmax(E1_model_fit)
    mean_model_E1 = (E1_grid[m_idx] + E1_grid[m_idx+1] / 2.0)

    E2_model_fit, E2_grid = kernel_fit_single(E2_model, params_list[case_id]["E2"]["model"]["bandwidth"])
    m_idx = np.argmax(E2_model_fit)
    mean_model_E2 = (E2_grid[m_idx] + E2_grid[m_idx+1]) / 2.0

    E3_model_fit, E3_grid = kernel_fit_single(E3_model, params_list[case_id]["E3"]["model"]["bandwidth"])
    m_idx = np.argmax(E3_model_fit)
    mean_model_E3 = (E3_grid[m_idx] + E3_grid[m_idx+1]) / 2.0

    E4_model_fit, E4_grid = kernel_fit_single(E4_model, params_list[case_id]["E4"]["model"]["bandwidth"])
    m_idx = np.argmax(E4_model_fit)
    mean_model_E4 = (E4_grid[m_idx] + E4_grid[m_idx+1]) / 2.0   

    
    count_list = add_grade(case_id, "post", eta_model, mean_model_clinical, E_post_PI.mean(), grade_list, count_old=None, n_grade=3)

    add_grade(case_id, "E", E_model, mean_model_E, E_PI.mean(), grade_list, count_old=count_list, n_grade=3)

    add_grade(case_id, "E1", E1_model, mean_model_E1, E1_PI.mean(), grade_list, count_old=count_list, n_grade=3)

    add_grade(case_id, "E2", E2_model, mean_model_E2, E2_PI.mean(), grade_list, count_old=count_list, n_grade=3)

    add_grade(case_id, "E3", E3_model, mean_model_E3, E3_PI.mean(), grade_list, count_old=count_list, n_grade=3)

    add_grade(case_id, "E4", E4_model, mean_model_E4, E4_PI.mean(), grade_list, count_old=count_list, n_grade=3)


    
    try:
        # Create target Directory
        test = os.path.join(write_file_dir, case_id, plots_dir)
        os.mkdir(test)
        print("Directory " , test ,  " Created ") 
    except FileExistsError:
        print("Directory " , test ,  " already exists")


process_grade = pd.DataFrame(grade_list, columns=pi_columns)
process_grade.to_pickle(process_grade_file)

with open(process_grade_file_list, 'wb') as f:
    pickle.dump(grade_list, f)

