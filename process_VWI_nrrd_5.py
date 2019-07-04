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


def bootstrap_resample(X, n=None, percent=0.01):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : numpy array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    percent: float, optional
      use a percentage of the data for the resample
    -------
    returns X_resamples based on percentage or number of samples
    defaults to the percentage
    p_n the number of sample used
    """
    if (n == None and percent == None):
        p_n = np.floor(0.01*X.shape[0])
    else:
        if ( n == None):
            p_n = np.floor(percent*X.shape[0]).astype(int)
        else:
            p_n = n
        
    #print(n, X.shape)
    X_resample = np.random.choice(X,p_n) # sampling with replacement
    return X_resample, p_n

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

def uncertainty(eta, xi,  conf=2.0, kind = 'E'):
    """
    use global variables to get what I want, this is pretty bad practice I think
    """

    if (kind == 'E'):
        u_E_2 = (u_eta_back_2 + u_xi_back_2 - 2.0 * u_eta_xi_back )
        u_E = np.sqrt(u_E_2)
        u_E_confidence = conf * u_E # gaussian 95% confidence interval
        return u_E_confidence, u_E, u_E_2
    elif(kind == 'E1'):
        u_E1_2 =  eta_vent_term + xi_vent_term + eta_term + xi_term + eta_xi_term + eta_xi_vent_term
        u_E1 = np.sqrt(u_E1_2)
        u_E1_confidence = conf * u_E1 # gaussian 95% confidence interval
        return u_E1_confidence, u_E1, u_E1_2
    elif (kind == 'E2'):
        u_E2_2 = (1.0 / (eta_vent**2.0) * u_eta_back_2 +
                        1.0 / (xi_vent**2.0) * u_xi_back_2 + 
                        np.square( eta / (eta_vent**2.0)) * u_eta_vent_2 +
                        np.square( xi / (xi_vent**2.0)) * u_xi_vent_2 -
                        2.0 / (eta_vent * xi_vent) * u_eta_xi_back -
                        2.0 * (eta * xi) / ( (eta_vent * xi_vent)**2.0 ) * u_eta_xi_vent
                        )
        u_E2 = np.sqrt(u_E2_2)
        u_E2_confidence = conf * u_E2 # gaussian 95% confidence interval
        
        return u_E2_confidence, u_E2, u_E2_2

    elif(kind == 'E3'):
        u_E3_2 = (1.0 / (u_eta_vent_2) * u_eta_back_2 +
                        1.0 / (u_xi_vent_2) * u_xi_back_2 -
                        2.0 / (np.sqrt(u_eta_vent_2 *u_xi_vent_2)) * u_eta_xi_back +
                        2.0 - 
                        2.0 / (np.sqrt(u_eta_vent_2 *u_xi_vent_2)) * u_eta_xi_vent
                        )
        u_E3 = np.sqrt(u_E3_2)
        u_E3_confidence = conf * u_E3 # gaussian 95% confidence interval

        return u_E3_confidence, u_E3, u_E3_2
    elif(kind == 'E4'):
        u_E4_2 = ( 1.0 / (u_eta_vent_2 + u_xi_vent_2) * ( 
                            u_eta_back_2 + u_xi_back_2 -
                            2.0 * u_eta_xi_back +
                            u_eta_vent_2 + u_xi_vent_2 -
                            2.0 * u_eta_xi_vent
                            )
                        )
        u_E4 = np.sqrt(u_E4_2)
        u_E4_confidence = conf * u_E4 # gaussian 95% confidence interval

        return u_E4_confidence, u_E4, u_E4_2
    else:
        raise  Exception("undefined enhancement kind {0}".format(kind))


def kernel_fit(data, min_size=20, cross_v=2, n_jobs_def= 22):
    """ guassian fit to 1D data
    """
    res = np.histogram(data.ravel(), bins='sqrt', density=True)
    std_data = data.std()
    bw = (data.ravel().shape[0] * (std_data+ 2) / 4.)**(-1. / (std_data + 4))
    bw_test = np.geomspace(bw/4.0, std_data, 20, endpoint=True)  # np.linspace(bw/2.0, std_data, 3) 

    
   
    N_bins = res[1].shape[0]
    if (N_bins < min_size):
        extra = 0.2
    else:
        extra = 0.0
    # get plus or minus 20%
    
    x_grid = np.linspace(res[1][0]-extra*abs(res[1][0]), res[1][-1] + extra*abs(res[1][0]), N_bins)
    
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bw_test},
                    cv=cross_v,
                    n_jobs=n_jobs_def) # 5-fold cross-validation
    
    #with joblib.parallel_backend('dask', n_jobs=n_jobs_def):
    grid.fit(data.ravel()[:, None])
    print("kernel fit: ", grid.best_params_)
    
    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    
    return pdf, x_grid, grid.best_params_


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

'''
A matplotlib-based function to overplot an elliptical error contour from the covariance matrix.
Copyright 2017 Megan Bedell (Flatiron).
Citations: Joe Kington (https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py),
           Vincent Spruyt (http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/)
'''
def error_ellipse(ax, xc, yc, cov, label, sigma=1, **kwargs):
    '''
    Plot an error ellipse contour over your data.
    Inputs:
    ax : matplotlib Axes() object
    xc : x-coordinate of ellipse center
    yc : x-coordinate of ellipse center
    cov : covariance matrix
    sigma : # sigma to plot (default 1)
    additional kwargs passed to matplotlib.patches.Ellipse()
    '''
    w, v = np.linalg.eigh(cov) # assumes symmetric matrix
    order = w.argsort()[::-1]
    w, v = w[order], v[:,order]
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    ellipse = patches.Ellipse(xy=(xc,yc),
                    width=2.*sigma*np.sqrt(w[0]),
                    height=2.*sigma*np.sqrt(w[1]),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    if (label != None):
        ellipse.set_label(label)
    ax.add_artist(ellipse)

def covariance_mat(A, B):
    return np.cov(np.vstack((A, B)))

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
pi_columns = ["Case ID", "Enhancement Type" , "Label", "Average"]
image_path_list = []
bootstrap_fig_list = []

case_id_list = []
e_type_list = []
label_list = []
average_list = []

uncertainty_list = []


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# create a plot of all the histograms
gs8 = plt.GridSpec(5,3, wspace=0.2, hspace=0.8)
fig8 = plt.figure(figsize=(13, 17))
ax1_8 = fig8.add_subplot(gs8[0, 0])
ax1_8.set_title("Compare Ventricle histograms", fontsize = font_size)
ax1_8.set_ylabel('Density, $E_0$', fontsize = font_size)
ax2_8 = fig8.add_subplot(gs8[1, 0])
ax2_8.set_ylabel('Density, $E_1$', fontsize = font_size)
ax3_8 = fig8.add_subplot(gs8[2, 0])
ax3_8.set_ylabel('Density $E_2$', fontsize = font_size)
ax4_8 = fig8.add_subplot(gs8[3, 0])
ax4_8.set_ylabel('Density  $E_3$', fontsize = font_size)
ax5_8 = fig8.add_subplot(gs8[4, 0])
ax5_8.set_xlabel('Intensity Values',   fontsize = font_size)
ax5_8.set_ylabel('Density $E_4$', fontsize = font_size)

ax6_8 = fig8.add_subplot(gs8[0, 1])
ax6_8.set_title("Compare Wall histograms", fontsize = font_size)
ax7_8 = fig8.add_subplot(gs8[1, 1])
ax8_8 = fig8.add_subplot(gs8[2, 1])
ax9_8 = fig8.add_subplot(gs8[3, 1])
ax10_8 = fig8.add_subplot(gs8[4, 1])
ax10_8.set_xlabel('Intensity Values',   fontsize = font_size)

ax11_8 = fig8.add_subplot(gs8[0, 2])
ax11_8.set_title("Compare Pituitary Infundibulum histograms", fontsize = font_size)
ax12_8 = fig8.add_subplot(gs8[1, 2])
ax13_8 = fig8.add_subplot(gs8[2, 2])
ax14_8 = fig8.add_subplot(gs8[3, 2])
ax15_8 = fig8.add_subplot(gs8[4, 2])
ax15_8.set_xlabel('Intensity Values',   fontsize = font_size)



# create a plot of all the pre vs post
gs6 = plt.GridSpec(6,3, wspace=0.2, hspace=0.6)
fig6 = plt.figure(figsize=(13, 17))
ax1_6 = fig6.add_subplot(gs6[0, 0])
ax1_6.set_title("Compare Ventricle Pre- vs. Post-VWI", fontsize = font_size)
ax1_6.set_ylabel(r'Post: $E_0$', fontsize = font_size)
ax1_6.set_xlabel(r'Pre: $E_0$', fontsize = font_size)
ax2_6 = fig6.add_subplot(gs6[1, 0])
ax2_6.set_ylabel(r'Post: $E_1$', fontsize = font_size)
ax2_6.set_xlabel(r'Pre: $E_1$', fontsize = font_size)
ax3_6 = fig6.add_subplot(gs6[2, 0])
ax3_6.set_ylabel(r'Post: $E_2$', fontsize = font_size)
ax3_6.set_xlabel(r'Pre: $E_2$', fontsize = font_size)
ax4_6 = fig6.add_subplot(gs6[3, 0])
ax4_6.set_ylabel(r'Post: $E_3$', fontsize = font_size)
ax4_6.set_xlabel(r'Pre: $E_3$', fontsize = font_size)
ax5_6 = fig6.add_subplot(gs6[4, 0])
ax5_6.set_ylabel(r'Post: $E_4$', fontsize = font_size)
ax5_6.set_xlabel(r'Pre: $E_4$',   fontsize = font_size)

ax6_6 = fig6.add_subplot(gs6[0, 1])
ax6_6.set_title("Compare Wall Pre- vs. Post-VWI", fontsize = font_size)
ax6_6.set_xlabel(r'Pre: $E_0$', fontsize = font_size)
ax7_6 = fig6.add_subplot(gs6[1, 1])
ax7_6.set_xlabel(r'Pre: $E_1$', fontsize = font_size)
ax8_6 = fig6.add_subplot(gs6[2, 1])
ax8_6.set_xlabel(r'Pre: $E_2$', fontsize = font_size)
ax9_6 = fig6.add_subplot(gs6[3, 1])
ax9_6.set_xlabel(r'Pre: $E_3$', fontsize = font_size)
ax10_6 = fig6.add_subplot(gs6[4, 1])
ax10_6.set_xlabel(r'Pre: $E_4$',   fontsize = font_size)

ax11_6 = fig6.add_subplot(gs6[0, 2])
ax11_6.set_title("Compare Pituitary Infundibulum Pre- vs. Post-VWI", fontsize = font_size)
ax11_6.set_xlabel(r'Pre: $E_0$', fontsize = font_size)
ax12_6 = fig6.add_subplot(gs6[1, 2])
ax12_6.set_xlabel(r'Pre: $E_1$', fontsize = font_size)
ax13_6 = fig6.add_subplot(gs6[2, 2])
ax13_6.set_xlabel(r'Pre: $E_2$', fontsize = font_size)
ax14_6 = fig6.add_subplot(gs6[3, 2])
ax14_6.set_xlabel(r'Pre: $E_3$', fontsize = font_size)
ax15_6 = fig6.add_subplot(gs6[4, 2])
ax15_6.set_xlabel(r'Pre: $E_4$',   fontsize = font_size)

# create a plot of all the pre vs post
gs5 = plt.GridSpec(5,3, wspace=0.2, hspace=0.5)
fig5 = plt.figure(figsize=(13, 17))
ax1_5 = fig5.add_subplot(gs5[0, 0])
ax1_5.set_title("Compare Ventricle Pre- vs. Post-VWI", fontsize = font_size)
ax1_5.set_ylabel(r'Post: $E_0$', fontsize = font_size)
ax1_5.set_xlabel(r'Pre: $E_0$', fontsize = font_size)
ax2_5 = fig5.add_subplot(gs5[1, 0])
ax2_5.set_ylabel(r'Post: $E_1$', fontsize = font_size)
ax2_5.set_xlabel(r'Pre: $E_1$', fontsize = font_size)
ax3_5 = fig5.add_subplot(gs5[2, 0])
ax3_5.set_ylabel(r'Post: $E_2$', fontsize = font_size)
ax3_5.set_xlabel(r'Pre: $E_2$', fontsize = font_size)
ax4_5 = fig5.add_subplot(gs5[3, 0])
ax4_5.set_ylabel(r'Post: $E_3$', fontsize = font_size)
ax4_5.set_xlabel(r'Pre: $E_3$', fontsize = font_size)
ax5_5 = fig5.add_subplot(gs5[4, 0])
ax5_5.set_ylabel(r'Post: $E_4$', fontsize = font_size)
ax5_5.set_xlabel(r'Pre: $E_4$',   fontsize = font_size)

ax6_5 = fig5.add_subplot(gs5[0, 1])
ax6_5.set_title("Compare Wall Pre- vs. Post-VWI", fontsize = font_size)
ax6_5.set_xlabel(r'Pre: $E_0$', fontsize = font_size)
ax7_5 = fig5.add_subplot(gs5[1, 1])
ax7_5.set_xlabel(r'Pre: $E_1$', fontsize = font_size)
ax8_5 = fig5.add_subplot(gs5[2, 1])
ax8_5.set_xlabel(r'Pre: $E_2$', fontsize = font_size)
ax9_5 = fig5.add_subplot(gs5[3, 1])
ax9_5.set_xlabel(r'Pre: $E_3$', fontsize = font_size)
ax10_5 = fig5.add_subplot(gs5[4, 1])
ax10_5.set_xlabel(r'Pre: $E_4$',   fontsize = font_size)

ax11_5 = fig5.add_subplot(gs5[0, 2])
ax11_5.set_title("Compare Pituitary Infundibulum Pre- vs. Post-VWI", fontsize = font_size)
ax11_5.set_xlabel(r'Pre: $E_0$', fontsize = font_size)
ax12_5 = fig5.add_subplot(gs5[1, 2])
ax12_5.set_xlabel(r'Pre: $E_1$', fontsize = font_size)
ax13_5 = fig5.add_subplot(gs5[2, 2])
ax13_5.set_xlabel(r'Pre: $E_2$', fontsize = font_size)
ax14_5 = fig5.add_subplot(gs5[3, 2])
ax14_5.set_xlabel(r'Pre: $E_3$', fontsize = font_size)
ax15_5 = fig5.add_subplot(gs5[4, 2])
ax15_5.set_xlabel(r'Pre: $E_4$',   fontsize = font_size)


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
    
    print(case_id, "post vent MEAN: {0:.4f} pre vent MEAN {1:.4f}".format(np.mean(img_post_vent), np.mean(img_pre_vent)))
    print(case_id, "post vent STD: {0:.4f} pre vent STD {1:.4f}".format(np.std(img_post_vent) , np.std(img_pre_vent) ))
    
    print(case_id, "post back MEAN: {0:.4f} pre back MEAN {1:.4f}".format(np.mean(img_post_back) , np.mean(img_pre_back) ))
    print(case_id, "post inter MEAN: {0:.4f} pre inter MEAN {1:.4f}".format(np.mean(img_post_back_inter) , np.mean(img_pre_back_inter) ))
    #print(case_id, "post vent inter shape: {0} pre vent inter shape {1}".format(img_post_back_inter.shape ,img_pre_back_inter.shape ))
    
    print(case_id, "post back STD: {0:.4f} pre back STD {1:.4f}".format(back_std_post, back_std_pre))
    print(case_id, "post inter STD: {0:.4f} pre inter STD {1:.4f}".format(np.std(img_post_back_inter) , np.std(img_pre_back_inter) ))
    
    print(case_id, "post PI Mean: {0:.4f} pre PI mean {1:.4f}".format(np.mean(eta_PI), np.mean(xi_PI)))
    print(case_id, "post PI STD: {0:.4f} pre PI STD {1:.4f}".format(np.std(eta_PI), np.std(xi_PI)))

    #koay_result_post, err = newton_koay(SNR_post_vent, channels)

    ## ventricle portion
    cov_vent = np.cov(np.vstack((img_post_vent, img_pre_vent)))
    #N_vent = float(img_post_vent.shape[0])

    #print(np.sqrt(2.0*np.trace(cov_vent) - np.sum(cov_vent)))

    eta_vent = np.mean(img_post_vent) # mean ventricle post
    xi_vent = np.mean(img_pre_vent) # mean ventricle pre
    u_eta_vent_2 = cov_vent[0,0] #/ N_vent  # uncertainty vent post square
    u_xi_vent_2 = cov_vent[1,1]  #/ N_vent  # uncertainty vent pre square
    u_eta_xi_vent = cov_vent[0,1] #/ N_vent # covariance between pre and post

    
    E = VWI_Enhancement(eta, xi, 1.0, 1.0, kind = "E1")
    case_id_list.append(case_id)
    e_type_list.append("E")
    label_list.append("Volume")
    average_list.append(E.mean())
    
    E1 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E1")
    case_id_list.append(case_id)
    e_type_list.append("E1")
    label_list.append("Volume")
    average_list.append(E1.mean())
    
    E2 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E2")
    case_id_list.append(case_id)
    e_type_list.append("E2")
    label_list.append("Volume")
    average_list.append(E2.mean())
    
    E3 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2))
    case_id_list.append(case_id)
    e_type_list.append("E3")
    label_list.append("Volume")
    average_list.append(E3.mean())
    
    E4 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2))
    case_id_list.append(case_id)
    e_type_list.append("E4")
    label_list.append("Volume")
    average_list.append(E4.mean())
    
    
    E_model, E_post_model, E_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                           1.0, 1.0, kind = "E1", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E")
    label_list.append("Wall")
    average_list.append(E_model.mean())
                        
    E1_model, E1_post_model, E1_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E1", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E1")
    label_list.append("Wall")
    average_list.append(E1_model.mean())
    
    E2_model, E2_post_model, E2_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E2", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E2")
    label_list.append("Wall")
    average_list.append(E2_model.mean())

    E3_model, E3_post_model, E3_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E3")
    label_list.append("Wall")
    average_list.append(E3_model.mean())
    
    E4_model, E4_post_model, E4_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E4")
    label_list.append("Wall")
    average_list.append(E4_model.mean())

    E_vent, E_post_vent, E_pre_vent = VWI_Enhancement(img_post_vent, img_pre_vent,
                                                           1.0, 1.0, kind = "E1", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E")
    label_list.append("Ventricle")
    average_list.append(E_vent.mean())

    
    E1_vent, E1_post_vent, E1_pre_vent  = VWI_Enhancement(img_post_vent, img_pre_vent,
                                                            eta_vent, xi_vent, kind = "E1", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E1")
    label_list.append("Ventricle")
    average_list.append(E1_vent.mean())

    E2_vent, E2_post_vent, E2_pre_vent  = VWI_Enhancement(img_post_vent, img_pre_vent,
                                                            eta_vent, xi_vent, kind = "E2", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E2")
    label_list.append("Ventricle")
    average_list.append(E2_vent.mean())

    E3_vent, E3_post_vent, E3_pre_vent = VWI_Enhancement(img_post_vent, img_pre_vent,
                                                            eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E3")
    label_list.append("Ventricle")
    average_list.append(E3_vent.mean())

    E4_vent, E4_post_vent, E4_pre_vent = VWI_Enhancement(img_post_vent, img_pre_vent,
                                                            eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E4")
    label_list.append("Ventricle")
    average_list.append(E4_vent.mean())


    E_PI, E_post_PI, E_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           1.0, 1.0, kind = "E1", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E")
    label_list.append("Pituitary Infundibulum")
    average_list.append(E_PI.mean())

    E1_PI, E1_post_PI, E1_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           eta_vent, xi_vent, kind = "E1", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E1")
    label_list.append("Pituitary Infundibulum")
    average_list.append(E1_PI.mean())
    E2_PI, E2_post_PI, E2_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           eta_vent, xi_vent, kind = "E2", return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E2")
    label_list.append("Pituitary Infundibulum")
    average_list.append(E2_PI.mean())
    E3_PI, E3_post_PI, E3_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                            eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E3")
    label_list.append("Pituitary Infundibulum")
    average_list.append(E3_PI.mean())

    E4_PI, E4_post_PI, E4_pre_PI  = VWI_Enhancement(eta_PI, xi_PI,
                                                            eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    case_id_list.append(case_id)
    e_type_list.append("E4")
    label_list.append("Pituitary Infundibulum")
    average_list.append(E4_PI.mean())

    cov_back = np.cov(np.vstack((img_post_back_inter, img_pre_back_inter)))

    #print(np.sqrt(2.0*np.trace(cov_vent) - np.sum(cov_vent)))
    u_eta_back_2 = cov_back[0,0] # uncertainty in post measures square
    u_xi_back_2 = cov_back[1,1] # uncertainty in pre measures square
    u_eta_xi_back = cov_back[0,1] # covariance estimate
    
    
    eta_vent_term = np.square((xi_vent / (eta_vent**2)) *  eta) * u_eta_vent_2
    xi_vent_term = np.square( eta / eta_vent) * u_xi_vent_2
    eta_term = np.square(xi_vent / eta_vent) * u_eta_back_2
    xi_term = u_xi_back_2

    eta_xi_term = -2.0 * xi_vent / eta_vent * u_eta_xi_back
    eta_xi_vent_term = -2.0 * xi_vent / (eta_vent ** 3.0) * np.square(eta) * u_eta_xi_vent

    # determine which term is the driver for uncertainty
    
    u_E_2,   u_E,   u_E_confidence   = uncertainty(eta, xi, conf, kind='E')
    u_E1_2, u_E1, u_E1_confidence = uncertainty(eta, xi, conf, kind='E1')
    u_E2_2, u_E2, u_E2_confidence = uncertainty(eta, xi, conf, kind='E2')
    u_E3_2, u_E3, u_E3_confidence = uncertainty(eta, xi, conf, kind='E3')
    u_E4_2, u_E4, u_E4_confidence = uncertainty(eta, xi, conf, kind='E4')
    uncertainty_list.extend([u_E.mean(), u_E1.mean(), u_E2.mean(), u_E3.mean(), u_E4.mean()])


    u_E_2_m,   u_E_m,   u_E_conf_m   = uncertainty(eta_model, xi_model, conf, kind='E')
    u_E1_2_m, u_E1_m, u_E1_conf_m = uncertainty(eta_model, xi_model, conf, kind='E1')
    u_E2_2_m, u_E2_m, u_E2_conf_m = uncertainty(eta_model, xi_model, conf, kind='E2')
    u_E3_2_m, u_E3_m, u_E3_conf_m = uncertainty(eta_model, xi_model, conf, kind='E3')
    u_E4_2_m, u_E4_m, u_E4_conf_m = uncertainty(eta_model, xi_model, conf, kind='E4')
    uncertainty_list.extend([u_E_m.mean(), u_E1_m.mean(), u_E2_m.mean(), u_E3_m.mean(), u_E4_m.mean()])
    
    u_E_2_v,   u_E_v,   u_E_conf_v   = uncertainty(eta_vent, xi_vent, conf, kind='E')
    u_E1_2_v, u_E1_v, u_E1_conf_v = uncertainty(eta_vent, xi_vent, conf, kind='E1')
    u_E2_2_v, u_E2_v, u_E2_conf_v = uncertainty(eta_vent, xi_vent, conf, kind='E2')
    u_E3_2_v, u_E3_v, u_E3_conf_v = uncertainty(eta_vent, xi_vent, conf, kind='E3')
    u_E4_2_v, u_E4_v, u_E4_conf_v = uncertainty(eta_vent, xi_vent, conf, kind='E4')
    uncertainty_list.extend([u_E_v.mean(), u_E1_v.mean(), u_E2_v.mean(), u_E3_v.mean(), u_E4_v.mean()])
    
    u_E_2_pi,   u_E_pi,   u_E_conf_pi   = uncertainty(eta_PI, xi_PI, conf, kind='E')
    u_E1_2_pi, u_E1_pi, u_E1_conf_pi = uncertainty(eta_PI, xi_PI, conf, kind='E1')
    u_E2_2_pi, u_E2_pi, u_E2_conf_pi = uncertainty(eta_PI, xi_PI, conf, kind='E2')
    u_E3_2_pi, u_E3_pi, u_E3_conf_pi = uncertainty(eta_PI, xi_PI, conf, kind='E3')
    u_E4_2_pi, u_E4_pi, u_E4_conf_pi = uncertainty(eta_PI, xi_PI, conf, kind='E4')
    uncertainty_list.extend([u_E_pi.mean(), u_E1_pi.mean(), u_E2_pi.mean(), u_E3_pi.mean(), u_E4_pi.mean()])


    case_id_list.append(case_id)
    e_type_list.append("Post")
    label_list.append("Wall")
    average_list.append(E_post_model.mean())
    case_id_list.append(case_id)
    e_type_list.append("Pre")
    label_list.append("Wall")
    average_list.append(E_pre_model.mean())
    
    case_id_list.append(case_id)
    e_type_list.append("Post")
    label_list.append("Ventricle")
    average_list.append(E_post_model.mean())
    case_id_list.append(case_id)
    e_type_list.append("Pre")
    label_list.append("Ventricle")
    average_list.append(E_pre_vent.mean())
    
    case_id_list.append(case_id)
    e_type_list.append("Post")
    label_list.append("Pituitary Infundibulum")
    average_list.append(E_post_PI.mean())
    case_id_list.append(case_id)
    e_type_list.append("Pre")
    label_list.append("Pituitary Infundibulum")
    average_list.append(E_pre_PI.mean())
    
    uncertainty_list.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    #u_E_2 = (u_eta_back_2 + u_xi_back_2 - 2.0 * u_eta_xi_back )
    #u_E = np.sqrt(u_E_2)
    #u_E_confidence = conf * u_E # gaussian 95% confidence interval

    #u_E1_2 =  eta_vent_term + xi_vent_term + eta_term + xi_term + eta_xi_term + eta_xi_vent_term
    #u_E1 = np.sqrt(u_E1_2)
    #u_E1_confidence = conf * u_E1 # gaussian 95% confidence interval


    #u_E2_2 = (1.0 / (eta_vent**2.0) * u_eta_back_2 +
                    #1.0 / (xi_vent**2.0) * u_xi_back_2 + 
                    #np.square( eta / (eta_vent**2.0)) * u_eta_vent_2 +
                    #np.square( xi / (xi_vent**2.0)) * u_xi_vent_2 -
                    #2.0 / (eta_vent * xi_vent) * u_eta_xi_back -
                    #2.0 * (eta * xi) / ( (eta_vent * xi_vent)**2.0 ) * u_eta_xi_vent
                    #)
    #u_E2 = np.sqrt(u_E2_2)
    #u_E2_confidence = conf * u_E2 # gaussian 95% confidence interval

    #u_E3_2 = (1.0 / (u_eta_vent_2) * u_eta_back_2 +
                    #1.0 / (u_xi_vent_2) * u_xi_back_2 -
                    #2.0 / (np.sqrt(u_eta_vent_2 *u_xi_vent_2)) * u_eta_xi_back +
                    #2.0 - 
                    #2.0 / (np.sqrt(u_eta_vent_2 *u_xi_vent_2)) * u_eta_xi_vent
                    #)
    #u_E3 = np.sqrt(u_E3_2)
    #u_E3_confidence = conf * u_E3 # gaussian 95% confidence interval


    #u_E4_2 = ( 1.0 / (u_eta_vent_2 + u_xi_vent_2) * ( 
                        #u_eta_back_2 + u_xi_back_2 -
                        #2.0 * u_eta_xi_back +
                        #u_eta_vent_2 + u_xi_vent_2 -
                        #2.0 * u_eta_xi_vent
                        #)
                    #)
    #u_E4 = np.sqrt(u_E4_2)
    #u_E4_confidence = conf * u_E4 # gaussian 95% confidence interval



    try:
        # Create target Directory
        test = os.path.join(write_file_dir, case_id, plots_dir)
        os.mkdir(test)
        print("Directory " , test ,  " Created ") 
    except FileExistsError:
        print("Directory " , test ,  " already exists")

    alpha_4 = 1.0
    gs4 = plt.GridSpec(3,4, wspace=0.2, hspace=0.8)
    fig4 = plt.figure(figsize=(13, 15))
    fig4.suptitle(r'{0}:  pre: $\xi$  vs. post: $\eta$ comparison'.format(case_id), fontsize=font_size)
    ax1_4 = fig4.add_subplot(gs4[0, 1:3])
    ax1_4.set_title("Compare $E_0$ pre vs post", fontsize = font_size)
    ax1_4.scatter(E_pre_model.ravel(), E_post_model.ravel(), label="Wall", alpha=alpha_4)
    ax1_4.scatter(E_pre_vent.ravel(), E_post_vent.ravel(), label="Ventricle", alpha=alpha_4)
    ax1_4.scatter(E_pre_PI.ravel(), E_post_PI.ravel(), label="Pituitary Infundibulum", alpha=alpha_4)
    ax1_4.set_xlabel(r'pre $\xi$', fontsize = font_size)
    ax1_4.set_ylabel(r'post $\eta$', fontsize = font_size)
    ax1_4.legend()
    
    ax2_4 = fig4.add_subplot(gs4[1, 0:2])
    ax2_4.set_title("Compare $E_1$ pre vs post", fontsize = font_size)
    ax2_4.scatter(E1_pre_model.ravel(), E1_post_model.ravel(), label="Wall", alpha=alpha_4)
    ax2_4.scatter(E1_pre_vent.ravel(), E1_post_vent.ravel(), label="Ventricle", alpha=alpha_4)
    ax2_4.scatter(E1_pre_PI.ravel(), E1_post_PI.ravel(), label="Pituitary Infundibulum", alpha=alpha_4)
    ax2_4.set_xlabel(r'pre $\xi$', fontsize = font_size)
    ax2_4.set_ylabel(r'post $\frac{ \bar{\xi}_{vent}}{ \bar{\eta}_{vent}} \eta$', fontsize = font_size)
    ax2_4.legend()

    ax3_4 = fig4.add_subplot(gs4[1, 2:4])
    ax3_4.set_title("Compare $E_2$ pre vs post", fontsize = font_size)
    ax3_4.scatter(E2_pre_model.ravel(), E2_post_model.ravel(), label="Wall", alpha=alpha_4)
    ax3_4.scatter(E2_pre_vent.ravel(), E2_post_vent.ravel(), label="Ventricle", alpha=alpha_4)
    ax3_4.scatter(E2_pre_PI.ravel(), E2_post_PI.ravel(), label="Pituitary Infundibulum", alpha=alpha_4)
    ax3_4.set_xlabel(r'pre $\frac{\xi}{\bar{\xi}_{vent}}$', fontsize = font_size)
    ax3_4.set_ylabel(r'post $\frac{\eta}{\bar{\eta}_{vent}}$', fontsize = font_size)
    ax3_4.legend()
    
    ax4_4 = fig4.add_subplot(gs4[2, 0:2])
    ax4_4.set_title("Compare $E_3$ pre vs post", fontsize = font_size)
    ax4_4.scatter(E3_pre_model.ravel(), E3_post_model.ravel(), label="Wall", alpha=alpha_4)
    ax4_4.scatter(E3_pre_vent.ravel(), E3_post_vent.ravel(), label="Ventricle", alpha=alpha_4)
    ax4_4.scatter(E3_pre_PI.ravel(), E3_post_PI.ravel(), label="Pituitary Infundibulum", alpha=alpha_4)
    ax4_4.set_xlabel(r'pre $ \frac{\xi - \bar{\xi}_{vent}}{ \sigma_{ \xi_{vent} } }$', fontsize = font_size)
    ax4_4.set_ylabel(r'post $\frac{\eta - \bar{\eta}_{vent}}{\sigma_{ \eta_{vent} } }$', fontsize = font_size)
    ax4_4.legend()
    
    ax5_4 = fig4.add_subplot(gs4[2, 2:4])
    ax5_4.set_title("Compare $E_4$ pre vs post", fontsize = font_size)
    ax5_4.scatter(E4_pre_model.ravel(), E4_post_model.ravel(), label="Wall", alpha=alpha_4)
    ax5_4.scatter(E4_pre_vent.ravel(), E4_post_vent.ravel(), label="Ventricle", alpha=alpha_4)
    ax5_4.scatter(E4_pre_PI.ravel(), E4_post_PI.ravel(), label="Pituitary Infundibulum", alpha=alpha_4)
    ax5_4.set_xlabel(r'pre $ \frac{\xi - \bar{\xi}_{vent}}{ \sqrt{\sigma^2_{ \eta_{vent}} + \sigma^2_{ \xi_{vent} } } }$', fontsize = font_size)
    ax5_4.set_ylabel(r'post $\frac{\eta - \bar{\eta}_{vent}}{\sqrt{\sigma^2_{ \eta_{vent}} + \sigma^2_{ \xi_{vent} } } }$', fontsize = font_size)
    ax5_4.legend()
    
    path_E3_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Enhancement.png")
    fig4.savefig(path_E3_model, dpi=dpi_value)
    plt.close(fig4)
    del fig4
    
    alpha_4 = 0.1
    alpha_ellipse = 1.0
    #gs5 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    #fig5 = plt.figure(figsize=(13, 13))
    #ax1_5 = fig5.add_subplot(gs5[0, 0])
    #ax1_5.set_title("Compare Model post vs. pre", fontsize = font_size)
    #ax1_5.scatter(E_pre_model.ravel() / E_pre_model.max(),
                  #E_post_model.ravel() / E_post_model.max(), label=r'$\frac{E}{max(E_0)}$', alpha=alpha_4)
    #ax1_5.scatter(E1_pre_model.ravel() / E1_pre_model.max(),
                  #E1_post_model.ravel() / E1_post_model.max(), label=r'$\frac{E_1}{max(E_1)}$', alpha=alpha_4)
    #ax1_5.scatter(E2_pre_model.ravel() / E2_pre_model.max(),
                  #E2_post_model.ravel() / E2_post_model.max(), label=r'$\frac{E_2}{max(E_2)}$', alpha=alpha_4)
    #ax1_5.scatter(E3_pre_model.ravel() / E3_pre_model.max(),
                  #E3_post_model.ravel() / E3_post_model.max(), label=r'$\frac{E_3}{max(E_3)}$', alpha=alpha_4)
    #ax1_5.scatter(E4_pre_model.ravel() / E4_pre_model.max(),
                  #E4_post_model.ravel() / E4_post_model.max(), label=r'$\frac{E_4}{max(E_4)}$', alpha=alpha_4)
    #ax1_5.set_xlabel(r'Pre: $E$', fontsize = font_size)
    #ax1_5.set_ylabel(r'Post: $E$', fontsize = font_size)
    #ax1_5.legend()
    
    #path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Model_normed.png")
    #fig5.savefig(path_ventricle_model, dpi=dpi_value)
    #plt.close(fig5)
    #del fig5
    
    #gs6 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    #fig6 = plt.figure(figsize=(13, 13))
    #ax1_6 = fig6.add_subplot(gs6[0, 0])
    #ax1_6.set_title("Compare Pituitary Infundibulum post vs. pre", fontsize = font_size)
    #ax1_6.scatter(E_pre_PI.ravel() / E_pre_PI.max(),
                  #E_post_PI.ravel()/E_post_PI.max(), label=r'$\frac{E}{max(E_0)}$', alpha=alpha_4)
    #ax1_6.scatter(E1_pre_PI.ravel()/E1_pre_PI.max(),
                  #E1_post_PI.ravel()/E1_post_PI.max(), label=r'$\frac{E_1}{max(E_1)}$', alpha=alpha_4)
    #ax1_6.scatter(E2_pre_PI.ravel() /E2_pre_PI.max(),
                  #E2_post_PI.ravel() / E2_post_PI.max(), label=r'$\frac{E_2}{max(E_2)}$', alpha=alpha_4)
    #ax1_6.scatter(E3_pre_PI.ravel() / E3_pre_PI.max(),
                  #E3_post_PI.ravel() / E3_post_PI.max(), label=r'$\frac{E_3}{max(E_3)}$', alpha=alpha_4)
    #ax1_6.scatter(E4_pre_PI.ravel() / E4_pre_PI.max(),
                  #E4_post_PI.ravel() / E4_post_PI.max(), label=r'$\frac{E_4}{max(E_4)}$', alpha=alpha_4)
    #ax1_6.set_xlabel(r'Pre: $E$', fontsize = font_size)
    #ax1_6.set_ylabel(r'Post: $E$', fontsize = font_size)
    #ax1_6.legend()
    
    #path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_PI_normed.png")
    #fig6.savefig(path_ventricle_model, dpi=dpi_value)
    #plt.close(fig6)
    #del fig6

    #gs7 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    #fig7 = plt.figure(figsize=(13, 13))
    #ax1_7 = fig7.add_subplot(gs7[0, 0])
    #ax1_7.set_title("Compare Ventricle post vs. pre ", fontsize = font_size)
    #ax1_7.scatter(E_pre_vent.ravel() / E_pre_vent.max(),
                  #E_post_vent.ravel() / E_post_vent.max(),   label=r'$\frac{E}{max(E_0)}$', alpha=alpha_4)
    #ax1_7.scatter(E1_pre_vent.ravel() / E1_pre_vent.max(),
                  #E1_post_vent.ravel() / E1_post_vent.max(), label=r'$\frac{E_1}{max(E_1)}$', alpha=alpha_4)
    #ax1_7.scatter(E2_pre_vent.ravel() / E2_pre_vent.max(),
                  #E2_post_vent.ravel() / E2_post_vent.max(), label=r'$\frac{E_2}{max(E_2)}$', alpha=alpha_4)
    #ax1_7.scatter(E3_pre_vent.ravel() / E3_pre_vent.max(),
                  #E3_post_vent.ravel() / E3_post_vent.max(), label=r'$\frac{E_3}{max(E_3)}$', alpha=alpha_4)
    #ax1_7.scatter(E4_pre_vent.ravel() / E4_pre_vent.max(),
                  #E4_post_vent.ravel() / E4_post_vent.max(), label=r'$\frac{E_4}{max(E_4)}$', alpha=alpha_4)
    #ax1_7.set_xlabel(r'Pre: $E$',   fontsize = font_size)
    #ax1_7.set_ylabel(r'Post: $E$', fontsize = font_size)
    #ax1_7.legend()
    
    #path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Ventricle_normed.png")
    #fig7.savefig(path_ventricle_model, dpi=dpi_value)
    #plt.close(fig7)
    #del fig7
    
    
    sc = ax1_5.scatter(E_pre_vent.ravel(), E_post_vent.ravel(),  alpha=alpha_4, label="{0}".format(case_id))
    sc_color = sc.get_facecolors()[0].tolist()
    ax2_5.scatter(E1_pre_vent.ravel(), E1_post_vent.ravel() ,   alpha=alpha_4, color=sc_color, label="{0}".format(case_id))
    ax3_5.scatter(E2_pre_vent.ravel(), E2_post_vent.ravel() ,   alpha=alpha_4, color=sc_color, label="{0}".format(case_id))
    ax4_5.scatter(E3_pre_vent.ravel(), E3_post_vent.ravel() ,   alpha=alpha_4, color=sc_color, label="{0}".format(case_id))
    ax5_5.scatter(E4_pre_vent.ravel(), E4_post_vent.ravel() ,   alpha=alpha_4, color=sc_color, label="{0}".format(case_id))

    sc2 =ax6_5.scatter(E_pre_model.ravel(), E_post_model.ravel() ,   alpha=alpha_4, label="{0}".format(case_id))
    sc_color2 = sc2.get_facecolors()[0].tolist()
    ax7_5.scatter(E1_pre_model.ravel(), E1_post_model.ravel() ,   alpha=alpha_4, color=sc_color2, label="{0}".format(case_id))
    ax8_5.scatter(E2_pre_model.ravel(), E2_post_model.ravel() ,   alpha=alpha_4, color=sc_color2, label="{0}".format(case_id))
    ax9_5.scatter(E3_pre_model.ravel(), E3_post_model.ravel() ,   alpha=alpha_4, color=sc_color2, label="{0}".format(case_id))
    ax10_5.scatter(E4_pre_model.ravel(), E4_post_model.ravel() ,   alpha=alpha_4, color=sc_color2, label="{0}".format(case_id))

    sc3 =ax11_5.scatter(E_pre_PI.ravel(), E_post_PI.ravel() ,   alpha=alpha_4, label="{0}".format(case_id))
    sc_color3 = sc3.get_facecolors()[0].tolist()
    ax12_5.scatter(E1_pre_PI.ravel(), E1_post_PI.ravel() ,   alpha=alpha_4, color=sc_color3, label="{0}".format(case_id))
    ax13_5.scatter(E2_pre_PI.ravel(), E2_post_PI.ravel() ,   alpha=alpha_4, color=sc_color3, label="{0}".format(case_id))
    ax14_5.scatter(E3_pre_PI.ravel(), E3_post_PI.ravel() ,   alpha=alpha_4, color=sc_color3, label="{0}".format(case_id))
    ax15_5.scatter(E4_pre_PI.ravel(), E4_post_PI.ravel() ,   alpha=alpha_4, color=sc_color3, label="{0}".format(case_id))
    
    error_ellipse(ax1_5, E_pre_vent.mean(), E_post_vent.mean(),
                  covariance_mat(E_pre_vent.ravel(), E_post_vent.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color)
    error_ellipse(ax2_5, E1_pre_vent.mean(), E1_post_vent.mean(),
                  covariance_mat(E1_pre_vent.ravel(), E1_post_vent.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color)
    error_ellipse(ax3_5, E2_pre_vent.mean(), E2_post_vent.mean(),
                  covariance_mat(E2_pre_vent.ravel(), E2_post_vent.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color)
    error_ellipse(ax4_5, E3_pre_vent.mean(), E3_post_vent.mean(),
                  covariance_mat(E3_pre_vent.ravel(), E3_post_vent.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color)

    error_ellipse(ax5_5, E4_pre_vent.mean(), E4_post_vent.mean(),
                  covariance_mat(E4_pre_vent.ravel(), E4_post_vent.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color)
    error_ellipse(ax6_5, E_pre_model.mean(), E_post_model.mean(),
                  covariance_mat(E_pre_model.ravel(), E_post_model.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color2)
    error_ellipse(ax7_5, E1_pre_model.mean(), E1_post_model.mean(),
                  covariance_mat(E1_pre_model.ravel(), E1_post_model.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color2)
    error_ellipse(ax8_5, E2_pre_model.mean(), E2_post_model.mean(),
                  covariance_mat(E2_pre_model.ravel(), E2_post_model.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color2)
    error_ellipse(ax9_5, E3_pre_model.mean(), E3_post_model.mean(),
                  covariance_mat(E3_pre_model.ravel(), E3_post_model.ravel()),None,
                  alpha=alpha_ellipse, ec = sc_color2)
    error_ellipse(ax10_5, E4_pre_model.mean(), E4_post_model.mean(),
                  covariance_mat(E4_pre_model.ravel(), E4_post_model.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color2)

    error_ellipse(ax11_5, E_pre_PI.mean(), E_post_PI.mean(),
                  covariance_mat(E_pre_PI.ravel(), E_post_PI.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color3)
    error_ellipse(ax12_5, E1_pre_PI.mean(), E1_post_PI.mean(),
                  covariance_mat(E1_pre_PI.ravel(), E1_post_PI.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color3)
    error_ellipse(ax13_5, E2_pre_PI.mean(), E2_post_PI.mean(),
                  covariance_mat(E2_pre_PI.ravel(), E2_post_PI.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color3)
    error_ellipse(ax14_5, E3_pre_PI.mean(), E3_post_PI.mean(),
                  covariance_mat(E3_pre_PI.ravel(), E3_post_PI.ravel()), None,
                  alpha=alpha_ellipse, ec = sc_color3)
    error_ellipse(ax15_5, E4_pre_PI.mean(), E4_post_PI.mean(),
                  covariance_mat(E4_pre_PI.ravel(), E4_post_PI.ravel()),None,
                  alpha=alpha_ellipse, ec = sc_color3)


    error_ellipse(ax1_6, E_pre_vent.mean(), E_post_vent.mean(),
                  covariance_mat(E_pre_vent.ravel(), E_post_vent.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color)
    ax1_6.set_xlim(ax1_5.get_xlim())
    ax1_6.set_ylim(ax1_5.get_ylim())
    error_ellipse(ax2_6, E1_pre_vent.mean(), E1_post_vent.mean(),
                  covariance_mat(E1_pre_vent.ravel(), E1_post_vent.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color)
    ax2_6.set_xlim(ax2_5.get_xlim())
    ax2_6.set_ylim(ax2_5.get_ylim())
    error_ellipse(ax3_6, E2_pre_vent.mean(), E2_post_vent.mean(),
                  covariance_mat(E2_pre_vent.ravel(), E2_post_vent.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color)
    ax3_6.set_xlim(ax3_5.get_xlim())
    ax3_6.set_ylim(ax3_5.get_ylim())
    error_ellipse(ax4_6, E3_pre_vent.mean(), E3_post_vent.mean(),
                  covariance_mat(E3_pre_vent.ravel(), E3_post_vent.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color)
    ax4_6.set_xlim(ax4_5.get_xlim())
    ax4_6.set_ylim(ax4_5.get_ylim())

    error_ellipse(ax5_6, E4_pre_vent.mean(), E4_post_vent.mean(),
                  covariance_mat(E4_pre_vent.ravel(), E4_post_vent.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color)
    ax5_6.set_xlim(ax5_5.get_xlim())
    ax5_6.set_ylim(ax5_5.get_ylim())
    error_ellipse(ax6_6, E_pre_model.mean(), E_post_model.mean(),
                  covariance_mat(E_pre_model.ravel(), E_post_model.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color2)
    ax6_6.set_xlim(ax6_5.get_xlim())
    ax6_6.set_ylim(ax6_5.get_ylim())
    error_ellipse(ax7_6, E1_pre_model.mean(), E1_post_model.mean(),
                  covariance_mat(E1_pre_model.ravel(), E1_post_model.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color2)
    ax7_6.set_xlim(ax7_5.get_xlim())
    ax7_6.set_ylim(ax7_5.get_ylim())
    error_ellipse(ax8_6, E2_pre_model.mean(), E2_post_model.mean(),
                  covariance_mat(E2_pre_model.ravel(), E2_post_model.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color2)
    ax8_6.set_xlim(ax8_5.get_xlim())
    ax8_6.set_ylim(ax8_5.get_ylim())
    error_ellipse(ax9_6, E3_pre_model.mean(), E3_post_model.mean(),
                  covariance_mat(E3_pre_model.ravel(), E3_post_model.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color2)
    ax9_6.set_xlim(ax9_5.get_xlim())
    ax9_6.set_ylim(ax9_5.get_ylim())
    error_ellipse(ax10_6, E4_pre_model.mean(), E4_post_model.mean(),
                  covariance_mat(E4_pre_model.ravel(), E4_post_model.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color2)
    ax10_6.set_xlim(ax10_5.get_xlim())
    ax10_6.set_ylim(ax10_5.get_ylim())
    error_ellipse(ax11_6, E_pre_PI.mean(), E_post_PI.mean(),
                  covariance_mat(E_pre_PI.ravel(), E_post_PI.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color3)
    ax11_6.set_xlim(ax11_5.get_xlim())
    ax11_6.set_ylim(ax11_5.get_ylim())
    error_ellipse(ax12_6, E1_pre_PI.mean(), E1_post_PI.mean(),
                  covariance_mat(E1_pre_PI.ravel(), E1_post_PI.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color3)
    ax12_6.set_xlim(ax12_5.get_xlim())
    ax12_6.set_ylim(ax12_5.get_ylim())
    error_ellipse(ax13_6, E2_pre_PI.mean(), E2_post_PI.mean(),
                  covariance_mat(E2_pre_PI.ravel(), E2_post_PI.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color3)
    ax13_6.set_xlim(ax13_5.get_xlim())
    ax13_6.set_ylim(ax13_5.get_ylim())
    error_ellipse(ax14_6, E3_pre_PI.mean(), E3_post_PI.mean(),
                  covariance_mat(E3_pre_PI.ravel(), E3_post_PI.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color3)
    ax14_6.set_xlim(ax14_5.get_xlim())
    ax14_6.set_ylim(ax14_5.get_ylim())
    error_ellipse(ax15_6, E4_pre_PI.mean(), E4_post_PI.mean(),
                  covariance_mat(E4_pre_PI.ravel(), E4_post_PI.ravel()), case_id,
                  alpha=alpha_ellipse, ec = sc_color3)
    ax15_6.set_xlim(ax15_5.get_xlim())
    ax15_6.set_ylim(ax15_5.get_ylim())


    #gs5 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    #fig5 = plt.figure(figsize=(13, 13))
    #ax1_5 = fig5.add_subplot(gs5[0, 0])
    #ax1_5.set_title("Compare Wall post vs. pre", fontsize = font_size)
    #ax1_5.scatter(E_pre_model.ravel(), E_post_model.ravel() , label=r'$E_0$', alpha=alpha_4)
    #ax1_5.scatter(E1_pre_model.ravel(), E1_post_model.ravel(), label=r'$E_1$', alpha=alpha_4)
    #ax1_5.scatter(E2_pre_model.ravel(), E2_post_model.ravel(), label=r'$E_2$', alpha=alpha_4)
    #ax1_5.scatter(E3_pre_model.ravel(), E3_post_model.ravel(), label=r'$E_3$', alpha=alpha_4)
    #ax1_5.scatter(E4_pre_model.ravel(), E4_post_model.ravel(), label=r'$E_4$', alpha=alpha_4)
    #ax1_5.set_xlabel(r'Pre: $E$', fontsize = font_size)
    #ax1_5.set_ylabel(r'Post: $E$', fontsize = font_size)
    #ax1_5.legend()
    
    #path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Model_data.png")
    #fig5.savefig(path_ventricle_model, dpi=dpi_value)
    #plt.close(fig5)
    #del fig5
    
    #gs6 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    #fig6 = plt.figure(figsize=(13, 13))
    #ax1_6 = fig6.add_subplot(gs6[0, 0])
    #ax1_6.set_title("Compare Pituitary Infundibulum post vs. pre", fontsize = font_size)
    #ax1_6.scatter(E_pre_PI.ravel(), E_post_PI.ravel(), label=r'$E_0$', alpha=alpha_4)
    #ax1_6.scatter(E1_pre_PI.ravel(), E1_post_PI.ravel(), label=r'$E_1$', alpha=alpha_4)
    #ax1_6.scatter(E2_pre_PI.ravel(), E2_post_PI.ravel(), label=r'$E_2$', alpha=alpha_4)
    #ax1_6.scatter(E3_pre_PI.ravel(), E3_post_PI.ravel(), label=r'$E_3$', alpha=alpha_4)
    #ax1_6.scatter(E4_pre_PI.ravel(), E4_post_PI.ravel(), label=r'$E_4$', alpha=alpha_4)
    #ax1_6.set_xlabel(r'Pre: $E$', fontsize = font_size)
    #ax1_6.set_ylabel(r'Post: $E$', fontsize = font_size)
    #ax1_6.legend()
    
    #path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_PI_data.png")
    #fig6.savefig(path_ventricle_model, dpi=dpi_value)
    #plt.close(fig6)
    #del fig6

    #gs7 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    #fig7 = plt.figure(figsize=(13, 13))
    #ax1_7 = fig7.add_subplot(gs7[0, 0])
    #ax1_7.set_title("Compare Ventricle post vs. pre ", fontsize = font_size)
    #ax1_7.scatter(E_pre_vent.ravel(), E_post_vent.ravel(),   label=r'$E_0$', alpha=alpha_4)
    #ax1_7.scatter(E1_pre_vent.ravel(), E1_post_vent.ravel(), label=r'$E_1$', alpha=alpha_4)
    #ax1_7.scatter(E2_pre_vent.ravel(), E2_post_vent.ravel(), label=r'$E_2$', alpha=alpha_4)
    #ax1_7.scatter(E3_pre_vent.ravel(), E3_post_vent.ravel(), label=r'$E_3$', alpha=alpha_4)
    #ax1_7.scatter(E4_pre_vent.ravel(),E4_post_vent.ravel(), label=r'$E_4$', alpha=alpha_4)
    #ax1_7.set_xlabel(r'Pre: $E$',   fontsize = font_size)
    #ax1_7.set_ylabel(r'Post: $E$', fontsize = font_size)
    #ax1_7.legend()
    
    #path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Ventricle_data.png")
    #fig7.savefig(path_ventricle_model, dpi=dpi_value)
    #plt.close(fig7)
    #del fig7
    
    gsz = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    figz = plt.figure(figsize=(13, 13))
    ax1_z = figz.add_subplot(gsz[0, 0])
    ax1_z.set_title("Compare Histogram Ventricle for $E$ ", fontsize = font_size)
    
    # estimate bandwidth
    bw = (E_vent.ravel().shape[0] * (E_vent.std() + 2) / 4.)**(-1. / (E_vent.std() + 4))
    bw2 = (E_model.ravel().shape[0] * (E_model.std() + 2) / 4.)**(-1. / (E_model.std() + 4))
    bw3 = (E_PI.ravel().shape[0] * (E_PI.std() + 2) / 4.)**(-1. / (E_PI.std() + 4))
    print("bw: ", bw, bw2, bw3)
    #E_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(E_vent.ravel()[:, None])
    
    
    #res1 = ax1_z.hist(E_vent.ravel(), bins='sqrt', label="{0}:{1}".format(case_id, "Ventricle"), alpha=alpha_4, density=True, histtype="step")
    #res1 = np.histogram(E_vent.ravel(), bins='sqrt', density=True)
    #E_kde_res = E_kde.score_samples(res1[1][:, None])
    #res2 = ax1_z.fill_between(res1[1], np.exp(E_kde_res), alpha=alpha_4, label=r'$E_0$')
    
    ax1_z.hist(E_vent.ravel(), bins='sqrt',   label=r'$E_0$', alpha=alpha_4, density=True)
    ax1_z.hist(E1_vent.ravel(), bins='sqrt', label=r'$E_1$', alpha=alpha_4, density=True)
    ax1_z.hist(E2_vent.ravel(), bins='sqrt', label=r'$E_2$', alpha=alpha_4, density=True)
    ax1_z.hist(E3_vent.ravel(), bins='sqrt', label=r'$E_3$', alpha=alpha_4, density=True)
    ax1_z.hist(E4_vent.ravel(), bins='sqrt', label=r'$E_4$', alpha=alpha_4, density=True)
    #ax1_z.set_xlabel(r'Pre: $E$',   fontsize = font_size)
    #ax1_z.set_ylabel(r'Post: $E$', fontsize = font_size)
    #ylims = ax1_z.get_ylim()
    xlims = ax1_z.get_xlim()
    ax1_z.set_ylim([0.0, 0.3])
    ax1_z.set_xlim([0.2*xlims[0], 0.2*xlims[1]])
    ax1_z.legend()
    
    path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_smooth_hist_data.png")
    figz.savefig(path_ventricle_model, dpi=dpi_value)
    plt.close(figz)
    del figz
    

    
    alpha_5 = 0.8
    #params_list[case_id] = {}
    E_vent_fit, E_grid, E_vent_params = kernel_fit_single(E_vent, params_list[case_id]["E"]["vent"]["bandwidth"])
    res2 = ax1_8.plot(E_grid, np.exp(E_vent_fit), alpha=alpha_5, label="{0}".format(case_id))
    plot_color = res2[0].get_color()
    
    E1_vent_fit, E1_grid, E1_vent_params= kernel_fit_single(E1_vent, params_list[case_id]["E1"]["vent"]["bandwidth"])
    #params_list[case_id].update({"E1" : {"vent" : E1_vent_params}})
    res = ax2_8.plot(E1_grid, np.exp(E1_vent_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color)

    E2_vent_fit, E2_grid, E2_vent_params= kernel_fit_single(E2_vent, params_list[case_id]["E2"]["vent"]["bandwidth"])
    #params_list[case_id].update( {"E2" : {"vent" : E2_vent_params}})
    res = ax3_8.plot(E2_grid, np.exp(E2_vent_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color)
    
    E3_vent_fit, E3_grid, E3_vent_params= kernel_fit_single(E3_vent, params_list[case_id]["E3"]["vent"]["bandwidth"])
    #params_list[case_id].update( {"E3" : {"vent" : E3_vent_params}})
    res = ax4_8.plot(E3_grid, np.exp(E3_vent_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color)
    
    E4_vent_fit, E4_grid, E4_vent_params = kernel_fit_single(E4_vent, params_list[case_id]["E4"]["vent"]["bandwidth"])
    #params_list[case_id].update( {"E4" : {"vent" : E4_vent_params}})
    res = ax5_8.plot(E4_grid, np.exp(E4_vent_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color)
    

    E_model_fit, E_grid, E_model_params = kernel_fit_single(E_model, params_list[case_id]["E"]["model"]["bandwidth"])
    #params_list[case_id]["E"].update({'model' : E_vent_params}) 
    res3 = ax6_8.plot(E_grid, np.exp(E_model_fit), alpha=alpha_5, label="{0}".format(case_id))
    plot_color2= res3[0].get_color()
    
    E1_model_fit, E1_grid, E1_model_params = kernel_fit_single(E1_model, params_list[case_id]["E1"]["model"]["bandwidth"])
    #params_list[case_id]["E1"].update({"model" :  E1_model_params})
    res = ax7_8.plot(E1_grid, np.exp(E1_model_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color2)

    E2_model_fit, E2_grid, E2_model_params = kernel_fit_single(E2_model, params_list[case_id]["E2"]["model"]["bandwidth"])
    #params_list[case_id]["E2"].update({"model": E2_model_params})
    res = ax8_8.plot(E2_grid, np.exp(E2_model_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color2)
    
    E3_model_fit, E3_grid, E3_model_params = kernel_fit_single(E3_model, params_list[case_id]["E3"]["model"]["bandwidth"])
    #params_list[case_id]["E3"].update({"model": E3_model_params})
    res = ax9_8.plot(E3_grid, np.exp(E3_model_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color2)
    
    E4_model_fit, E4_grid, E4_model_params = kernel_fit_single(E4_model, params_list[case_id]["E4"]["model"]["bandwidth"])
    #params_list[case_id]["E4"].update({"model": E4_model_params})
    res = ax10_8.plot(E4_grid, np.exp(E4_model_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color2)
    
    
    E_PI_fit, E_grid, E_PI_params = kernel_fit_single(E_PI, params_list[case_id]["E"]["PI"]["bandwidth"])
    #params_list[case_id]["E"].update({"PI": E_PI_params})
    res4 = ax11_8.plot(E_grid, np.exp(E_PI_fit), alpha=alpha_5, label="{0}".format(case_id))
    plot_color3= res4[0].get_color()
    
    E1_PI_fit, E1_grid, E1_PI_params = kernel_fit_single(E1_PI, params_list[case_id]["E1"]["PI"]["bandwidth"])
    #params_list[case_id]["E1"].update({"PI" :  E1_PI_params})
    res = ax12_8.plot(E1_grid, np.exp(E1_PI_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color3)

    E2_PI_fit, E2_grid, E2_PI_params = kernel_fit_single(E2_PI, params_list[case_id]["E2"]["PI"]["bandwidth"])
    #params_list[case_id]["E2"].update( {"PI" :  E2_PI_params})
    res = ax13_8.plot(E2_grid, np.exp(E2_PI_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color3)
    
    E3_PI_fit, E3_grid, E3_PI_params = kernel_fit_single(E3_PI, params_list[case_id]["E3"]["PI"]["bandwidth"])
    #params_list[case_id]["E3"].update( {"PI" : E3_PI_params})
    res = ax14_8.plot(E3_grid, np.exp(E3_PI_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color3)
    
    E4_PI_fit, E4_grid, E4_PI_params = kernel_fit_single(E4_PI, params_list[case_id]["E4"]["PI"]["bandwidth"])
    #params_list[case_id]["E4"].update({"PI" : E4_PI_params})
    res = ax15_8.plot(E4_grid, np.exp(E4_PI_fit), alpha=alpha_5, label="{0}".format(case_id), color=plot_color3)

    
    
    #res1 = ax1_8.hist(E_vent.ravel(), bins='sqrt', label="{0}".format(case_id), alpha=alpha_5, density=True, histtype='step')
    #hist_color = res1[2][0].get_facecolor()
    #res = ax2_8.hist(E1_vent.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color, alpha=alpha_5, density=True, histtype='step')
    #res = ax3_8.hist(E2_vent.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color, alpha=alpha_5, density=True, histtype='step')
    #res = ax4_8.hist(E3_vent.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color, alpha=alpha_5, density=True, histtype='step')
    #res = ax5_8.hist(E4_vent.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color, alpha=alpha_5, density=True, histtype='step')
    
    
    #res2 = ax6_8.hist(E_model.ravel(), bins='sqrt', label="{0}".format(case_id), alpha=alpha_5, density=True, histtype='step')
    #hist_color2 = res2[2][0].get_facecolor()
    #res = ax7_8.hist(E1_model.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color2, alpha=alpha_5, density=True, histtype='step')
    #res = ax8_8.hist(E2_model.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color2, alpha=alpha_5, density=True, histtype='step')
    #res = ax9_8.hist(E3_model.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color2, alpha=alpha_5, density=True, histtype='step')
    #res = ax10_8.hist(E4_model.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color2, alpha=alpha_5, density=True, histtype='step')
    
    #res3 = ax11_8.hist(E_PI.ravel(), bins='sqrt', label="{0}".format(case_id), alpha=alpha_5, density=True, histtype='step')
    #hist_color3 = res3[2][0].get_facecolor()
    #res = ax12_8.hist(E1_PI.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color3, alpha=alpha_5, density=True, histtype='step')
    #res = ax13_8.hist(E2_PI.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color3, alpha=alpha_5, density=True, histtype='step')
    #res = ax14_8.hist(E3_PI.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color3, alpha=alpha_5, density=True, histtype='step')
    #res = ax15_8.hist(E4_PI.ravel(), bins='sqrt', label="{0}".format(case_id), color=hist_color3, alpha=alpha_5, density=True, histtype='step')

    
    
    # create histogram  plots of image and model
    n_bins2 = 4000
    n_bins3 = 100
    shrink_y = 0.2 # fraction of ylim to view distribution
    shrink_x = 0.2
    gs2 = plt.GridSpec(5,4, wspace=0.2, hspace=0.8)
    #gs2 = plt.GridSpec(4,4, wspace=0.2, hspace=0.8) 
    # Create a figure
    fig2 = plt.figure(figsize=(17, 19))

    # SUBFIGURE 1
    # Create subfigure 1 (takes over two rows (0 to 1) and column 0 of the grid)
    ax1a_2 = fig2.add_subplot(gs2[0, 0])
    ax1a_2.set_title("{0}: $E_0$  Volume".format(case_id), fontsize = font_size)
    ax1a_2.set_ylabel("count", fontsize = font_size)
    #ax1a_2.set_xlabel("Enhancement", fontsize = font_size)
    ax1b_2 = fig2.add_subplot(gs2[0,1])
    ax1b_2.set_title("{0}: $E_0$ Vessel Wall".format(case_id), fontsize = font_size)
    #ax1b_2.set_ylabel("count", fontsize = font_size)
    ax1c_2 = fig2.add_subplot(gs2[0, 2])
    ax1c_2.set_title("{0}: $E_0$ Ventricle".format(case_id), fontsize = font_size)
    #ax1b_2.set_ylabel("count", fontsize = font_size)
    ax1d_2 = fig2.add_subplot(gs2[0, 3])
    ax1d_2.set_title("{0}: $E_0$ Pituitary Infundibulum".format(case_id), fontsize = font_size)
    #ax1d_2.set_title(r'{0}: $\bar{U}_{{E}}$ Volume'.format(case_id), fontsize = font_size)


    ax2a_2 = fig2.add_subplot(gs2[1, 0])
    ax2a_2.set_title("{0}: $E_1$  Volume".format(case_id), fontsize = font_size)
    ax2a_2.set_ylabel("count", fontsize = font_size)
    ax2b_2 = fig2.add_subplot(gs2[1, 1])
    ax2b_2.set_title("{0}: $E_1$ Vessel Wall".format(case_id), fontsize = font_size)
    ax2c_2 = fig2.add_subplot(gs2[1, 2])
    ax2c_2.set_title("{0}: $E_1$ Ventricle".format(case_id), fontsize = font_size)
    ax2d_2 = fig2.add_subplot(gs2[1, 3])
    ax2d_2.set_title("{0}: $E_1$ Pituitary Infundibulum".format(case_id), fontsize = font_size)
    #ax2d_2.set_title(r'{0}: $\bar{U}_{{E_1}}$ Volume'.format(case_id), fontsize = font_size)
    
    ax3a_2 = fig2.add_subplot(gs2[2, 0])
    ax3a_2.set_title("{0}: $E_2$  Volume".format(case_id), fontsize = font_size)
    ax3a_2.set_ylabel("count", fontsize = font_size)
    ax3b_2 = fig2.add_subplot(gs2[2, 1])
    ax3b_2.set_title("{0}: $E_2$ Vessel Wall".format(case_id), fontsize = font_size)
    ax3c_2 = fig2.add_subplot(gs2[2, 2])
    ax3c_2.set_title("{0}: $E_2$ Ventricle".format(case_id), fontsize = font_size)
    ax3d_2 = fig2.add_subplot(gs2[2, 3])
    ax3d_2.set_title("{0}: $E_2$ Pituitary Infundibulum".format(case_id), fontsize = font_size)
    #ax3d_2.set_title(r'{0}: $\bar{U}_{{E_2}}$ Volume'.format(case_id), fontsize = font_size)
    
    ax4a_2 = fig2.add_subplot(gs2[3, 0])
    ax4a_2.set_title("{0}: $E_3$  Volume".format(case_id), fontsize = font_size)
    ax4a_2.set_ylabel("count", fontsize = font_size)
    ax4b_2 = fig2.add_subplot(gs2[3, 1])
    ax4b_2.set_title("{0}: $E_3$ Vessel Wall".format(case_id), fontsize = font_size)
    ax4c_2 = fig2.add_subplot(gs2[3, 2])
    ax4c_2.set_title("{0}: $E_3$ Ventricle".format(case_id), fontsize = font_size)
    ax4d_2 = fig2.add_subplot(gs2[3, 3])
    ax4d_2.set_title("{0}: $E_3$ Pituitary Infundibulum".format(case_id), fontsize = font_size)
    #ax4d_2.set_title(r'{0}: $\bar{U}_{{E_3}}$ Volume'.format(case_id), fontsize = font_size)

    ax5a_2 = fig2.add_subplot(gs2[4, 0])
    ax5a_2.set_title("{0}: $E_4$  Volume".format(case_id), fontsize = font_size)
    ax5a_2.set_ylabel("count", fontsize = font_size)
    ax5a_2.set_xlabel("Enhancement", fontsize = font_size)    
    ax5b_2 = fig2.add_subplot(gs2[4, 1])
    ax5b_2.set_title("{0}: $E_4$  Vessel Wall".format(case_id), fontsize = font_size)
    ax5b_2.set_xlabel("Enhancement", fontsize = font_size)
    ax5c_2 = fig2.add_subplot(gs2[4, 2])
    ax5c_2.set_title("{0}: $E_4$ Ventricle".format(case_id), fontsize = font_size)
    ax5c_2.set_xlabel("Enhancement", fontsize = font_size)
    ax5d_2 = fig2.add_subplot(gs2[4, 3])
    ax5d_2.set_title("{0}: $E_4$ Pituitary Infundibulum".format(case_id), fontsize = font_size)
    #ax5d_2.set_title(r'{0}: $\bar{U}_{{E_4}}$ Volume'.format(case_id), fontsize = font_size)

    ax1a_2.hist(E.ravel(), bins='auto', label="$E_0$")
    ax1a_2.axvline(x=np.mean(E), color='r', label="mean")
    ax1a_2.axvline(x=u_E.mean(), color='k', label=r'$+\bar{U}_{E_0}$')
    #ax1a_2.axvline(x=-u_E.mean(), color='k', label=r'$-\bar{U}_{E_0}$')
    ymin, ymax = ax1a_2.get_ylim()
    ax1a_2.set_ylim([ymin, shrink_y*ymax])
    xmin, xmax = ax1a_2.get_xlim()
    ax1a_2.set_xlim([shrink_x*xmin, shrink_x*xmax])
    ax1a_2.legend()
    ax1b_2.hist(E_model.ravel(), bins='auto', label="$E_0$")
    ax1b_2.axvline(x=np.mean(E_model), color='r', label="mean")
    ax1b_2.axvline(x=u_E_m.mean(), color='k', label=r'$+\bar{U}_{E_0}$')
    #ax1b_2.axvline(x=-u_E_m.mean(), color='k', label=r'$-\bar{U}_{E_0}$')
    ax1b_2.legend()
    ax1c_2.hist(E_vent.ravel(), bins='auto', label="$E_0$")
    ax1c_2.axvline(x=np.mean(E_vent), color='r', label="mean")
    ax1c_2.axvline(x=u_E_v.mean(), color='k', label=r'$+\bar{U}_{E_0}$')
    #ax1c_2.axvline(x=-u_E_v.mean(), color='k', label=r'$-\bar{U}_{E_0}$')
    ax1c_2.legend()
    ax1d_2.hist(E_PI.ravel(), bins='auto', label="$E_0$")
    ax1d_2.axvline(x=np.mean(E_PI), color='r', label="mean")
    ax1d_2.axvline(x=u_E_pi.mean(), color='k', label=r'$+\bar{U}_{E_0}$')
    #ax1d_2.axvline(x=-u_E_pi.mean(), color='k', label=r'$-\bar{U}_{E_0}$')
    ax1d_2.legend()
    #ax1d_2.hist(u_E.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E}$ Volume", density=True)
    #ax1d_2.hist(u_E_m.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E}$ Wall", density=True)
    #ax1d_2.hist(u_E_v.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E}$ Ventricle", density=True)
    #ax1d_2.legend()
    
    ax2a_2.hist(E1.ravel(), bins='auto', label="$E_1$")
    ax2a_2.axvline(x=np.mean(E), color='r', label="mean")
    ax2a_2.axvline(x=u_E1.mean(), color='k', label=r'$+\bar{U}_{E_1}$')
    #ax2a_2.axvline(x=-u_E1.mean(), color='k', label=r'$-\bar{U}_{E_1}$')
    ymin, ymax = ax2a_2.get_ylim()
    ax2a_2.set_ylim([ymin, shrink_y*ymax])
    xmin, xmax = ax2a_2.get_xlim()
    ax2a_2.set_xlim([shrink_x*xmin, shrink_x*xmax])
    ax2a_2.legend()
    ax2b_2.hist(E1_model.ravel(), bins='auto', label="$E_1$")
    ax2b_2.axvline(x=np.mean(E1_model), color='r', label="mean")
    ax2b_2.axvline(x=u_E1_m.mean(), color='k', label=r'$+\bar{U}_{E_1}$')
    #ax2b_2.axvline(x=-u_E1_m.mean(), color='k', label=r'$-\bar{U}_{E_1}$')
    ax2b_2.legend()
    ax2c_2.hist(E1_vent.ravel(), bins='auto', label="$E_1$t")
    ax2c_2.axvline(x=np.mean(E1_vent), color='r', label="mean")
    ax2c_2.axvline(x=u_E1_v.mean(), color='k', label=r'$+\bar{U}_{E_1}$')
    #ax2c_2.axvline(x=-u_E1_v.mean(), color='k', label=r'$-\bar{U}_{E_1}$')
    ax2c_2.legend()
    ax2d_2.hist(E1_PI.ravel(), bins='auto', label="$E_1$")
    ax2d_2.axvline(x=np.mean(E1_PI), color='r', label="mean")
    ax2d_2.axvline(x=u_E1_pi.mean(), color='k', label=r'$+\bar{U}_{E_1}$')
    #ax2d_2.axvline(x=-u_E1_pi.mean(), color='k', label=r'$-\bar{U}_{E_1}$')
    ax2d_2.legend()
    #ax2d_2.hist(u_E1.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_1}$ Volume", density=True)
    #ax2d_2.hist(u_E1_m.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_1}$ Wall", density=True)
    #ax2d_2.hist(u_E1_v.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_1}$ Ventricle", density=True)
    #ax2d_2.legend()

    ax3a_2.hist(E2.ravel(), bins='auto', label="$E_2$")
    ax3a_2.axvline(x=np.mean(E2), color='r', label="mean")
    ax3a_2.axvline(x=u_E2.mean(), color='k', label=r'$+\bar{U}_{E_2}$')
    #ax3a_2.axvline(x=-u_E2.mean(), color='k', label=r'$-\bar{U}_{E_2}$')
    ymin, ymax = ax3a_2.get_ylim()
    ax3a_2.set_ylim([ymin, shrink_y*ymax])
    xmin, xmax = ax3a_2.get_xlim()
    ax3a_2.set_xlim([shrink_x*xmin, shrink_x*xmax])
    ax3a_2.legend()
    ax3b_2.hist(E2_model.ravel(), bins='auto', label="$E_2$")
    ax3b_2.axvline(x=np.mean(E2_model), color='r', label="mean")
    ax3b_2.axvline(x=u_E2_m.mean(), color='k', label=r'$+\bar{U}_{E_2}$')
    #ax3b_2.axvline(x=-u_E2_m.mean(), color='k', label=r'$-\bar{U}_{E_2}$')
    ax3b_2.legend()
    ax3c_2.hist(E2_vent.ravel(), bins='auto', label="$E_2$")
    ax3c_2.axvline(x=np.mean(E2_vent), color='r', label="mean")
    ax3c_2.axvline(x=u_E2_v.mean(), color='k', label=r'$+\bar{U}_{E_2}$')
    #ax3c_2.axvline(x=-u_E2_v.mean(), color='k', label=r'$-\bar{U}_{E_2}$')
    ax3c_2.legend()
    ax3d_2.hist(E2_PI.ravel(), bins='auto', label="$E_2$")
    ax3d_2.axvline(x=np.mean(E2_PI), color='r', label="mean")
    ax3d_2.axvline(x=u_E2_pi.mean(), color='k', label=r'$+\bar{U}_{E_2}$')
    #ax3d_2.axvline(x=-u_E2_pi.mean(), color='k', label=r'$-\bar{U}_{E_2}$')
    ax3d_2.legend()
    #ax3d_2.hist(u_E2.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_2}$ Volume", density=True)
    #ax3d_2.hist(u_E2_m.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_2}$ Wall", density=True)
    #ax3d_2.hist(u_E2_v.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_2}$ Ventricle", density=True)
    #ax3d_2.legend()

    ax4a_2.hist(E3.ravel(), bins='auto', label="$E_3$")
    ax4a_2.axvline(x=np.mean(E3), color='r', label="mean")
    ax4a_2.axvline(x=u_E3.mean(), color='k', label=r'$+\bar{U}_{E_3}$')
    #ax4a_2.axvline(x=-u_E3.mean(), color='k', label=r'$-\bar{U}_{E_3}$')
    ymin, ymax = ax4a_2.get_ylim()
    ax4a_2.set_ylim([ymin, shrink_y*ymax])
    xmin, xmax = ax4a_2.get_xlim()
    ax4a_2.set_xlim([shrink_x*xmin, shrink_x*xmax])
    ax4a_2.legend()
    ax4b_2.hist(E3_model.ravel(), bins='auto', label="$E_3$")
    ax4b_2.axvline(x=np.mean(E3_model), color='r', label="mean")
    ax4b_2.axvline(x=u_E3_m.mean(), color='k', label=r'$+\bar{U}_{E_3}$')
    #ax4b_2.axvline(x=-u_E3_m.mean(), color='k', label=r'$-\bar{U}_{E_3}$')
    ax4b_2.legend()
    ax4c_2.hist(E3_vent.ravel(), bins='auto', label="$E_1$")
    ax4c_2.axvline(x=np.mean(E3_vent), color='r', label="mean")
    ax4c_2.axvline(x=u_E3_v.mean(), color='k', label=r'$+\bar{U}_{E_3}$')
    #ax4c_2.axvline(x=-u_E3_v.mean(), color='k', label=r'$-\bar{U}_{E_3}$')
    ax4c_2.legend()
    ax4d_2.hist(E3_PI.ravel(), bins='auto', label="$E_3$")
    ax4d_2.axvline(x=np.mean(E3_PI), color='r', label="mean")
    ax4d_2.axvline(x=u_E3_pi.mean(), color='k', label=r'$+\bar{U}_{E_3}$')
    #ax4d_2.axvline(x=-u_E3_pi.mean(), color='k', label=r'$-\bar{U}_{E_3}$')
    ax4d_2.legend()
    #ax4d_2.hist(u_E3.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_3}$ Volume", density=True)
    #ax4d_2.hist(u_E3_m.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_3}$ Wall", density=True)
    #ax4d_2.hist(u_E3_v.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_3}$ Ventricle", density=True)
    #ax4d_2.legend()

    ax5a_2.hist(E4.ravel(), bins='auto', label="$E_4$")
    ax5a_2.axvline(x=np.mean(E4), color='r', label="mean")
    ax5a_2.axvline(x=u_E4.mean(), color='k', label=r'$+\bar{U}_{E_4}$')
    #ax5a_2.axvline(x=-u_E4.mean(), color='k', label=r'$-\bar{U}_{E_4}$')
    ymin, ymax = ax5a_2.get_ylim()
    ax5a_2.set_ylim([ymin, shrink_y*ymax])
    xmin, xmax = ax5a_2.get_xlim()
    ax5a_2.set_xlim([shrink_x*xmin, shrink_x*xmax])
    ax5a_2.legend()
    ax5b_2.hist(E4_model.ravel(), bins='auto', label="$E_4$ model")
    ax5b_2.axvline(x=np.mean(E4_model), color='r', label="mean")
    ax5b_2.axvline(x=u_E4_m.mean(), color='k', label=r'$+\bar{U}_{E_4}$')
    #ax5b_2.axvline(x=-u_E4_m.mean(), color='k', label=r'$-\bar{U}_{E_4}$')
    ax5b_2.legend()
    ax5c_2.hist(E4_vent.ravel(), bins='auto', label="$E_4$  vent")
    ax5c_2.axvline(x=np.mean(E4_vent), color='r', label="mean")
    ax5c_2.axvline(x=u_E4_v.mean(), color='k', label=r'$+\bar{U}_{E_4}$')
    #ax5c_2.axvline(x=-u_E4_v.mean(), color='k', label=r'$-\bar{U}_{E_4}$')
    ax5c_2.legend()
    ax5d_2.hist(E4_PI.ravel(), bins='auto', label="$E_4$")
    ax5d_2.axvline(x=np.mean(E4_PI), color='r', label="mean")
    ax5d_2.axvline(x=u_E4_pi.mean(), color='k', label=r'$+\bar{U}_{E_2}$')
    #ax5d_2.axvline(x=-u_E4_pi.mean(), color='k', label=r'$-\bar{U}_{E_2}$')
    ax5d_2.legend()
    #ax5d_2.hist(u_E4.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_4}$ Volume", density=True)
    #ax5d_2.hist(u_E4_m.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_4}$ Wall", density=True)
    #ax5d_2.hist(u_E4_v.ravel(), bins='auto', alpha=alpha_4, label="$\bar{U}_{E_4}$ Ventricle", density=True)
    #ax5d_2.legend()

    #ax8_2.hist(u_E_confidence.ravel(), bins=n_bins3)
    #ax8_2.axvline(x=np.mean(u_E_confidence), color='r')
    #ax8_2.set_ylabel("count", fontsize = font_size)
    path_images = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Enhancement_Distribution.png")
    
    image_path_list.append(path_images)
    fig2.savefig(path_images, dpi=dpi_value)
    plt.close(fig2)
    del fig2
    

   
    if (skip_write) :
        pass
    else:
 
        
        print(np.mean(u_E_confidence))
        print(np.mean(u_E1_confidence))
        print(np.mean(u_E2_confidence))
        print(np.mean(u_E3_confidence))
        print(np.mean(u_E4_confidence))
        print("pink")
        

        write_list = {}
        E_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], 1.0, 1.0, kind = "E1")
        write_list["E.nrrd"] = E_full
        E1_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E1")
        write_list["E1.nrrd"] = E1_full
        E2_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E2")
        write_list["E2.nrrd"] = E2_full
        E3_full= VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E3",
                                                            std_post_vent = np.sqrt(u_eta_vent_2), 
                                                            std_pre_vent = np.sqrt(u_xi_vent_2), 
                                                            return_parts=False)
        write_list["E3.nrrd"] = E3_full
        E4_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E4",
                                                            std_post_vent = np.sqrt(u_eta_vent_2), 
                                                            std_pre_vent = np.sqrt(u_xi_vent_2), 
                                                            return_parts=False)
        write_list["E4.nrrd"] = E4_full


        #gs3 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
        #fig3 = plt.figure(figsize=(13, 13))
        #ax1_3 = fig3.add_subplot(gs3[0, 0])
        #ax1_3.set_title("Compare E3 pre vs post", fontsize = font_size)
        #ax1_3.scatter(E3_pre.ravel(), E3_post.ravel())
        #ax1_3.set_xlabel(r'pre $ \frac{\xi - \bar{\xi}_{vent}}{std(\xi_{vent})}$', fontsize = font_size)
        #ax1_3.set_ylabel(r'post $\frac{\eta - \bar{\eta}_{vent}}{std(\eta_{vent})}$', fontsize = font_size)
        
        #path_E3 = os.path.join(write_file_dir, case_id, plots_dir, "Compare_E3.png")
        #fig3.savefig(path_E3)
        #del fig3
        
        #E_non-zeros = np.where(E == 0.0)
        
        E_term_frac   = np.divide(E, u_E_confidence.mean(), dtype=np.float)
        E1_term_frac = np.divide(E1, u_E1_confidence.mean(), dtype=np.float)
        E2_term_frac = np.divide(E2, u_E2_confidence.mean(), dtype=np.float)
        E3_term_frac = np.divide(E3,  u_E3_confidence.mean(), dtype=np.float)
        E4_term_frac = np.divide(E4,  u_E4_confidence.mean(), dtype=np.float)
        
        
        # assuming the null hypothesis is that the pixel value for E is zero
        P_E   = 1.0 - stats.norm.cdf( E_term_frac / conf )
        P_E1 = 1.0 - stats.norm.cdf( E1_term_frac / conf )
        P_E2 = 1.0 - stats.norm.cdf( E2_term_frac / conf)
        P_E3 = 1.0 - stats.norm.cdf( E3_term_frac / conf)
        P_E4 = 1.0 - stats.norm.cdf( E4_term_frac / conf)
            
        
        # create confidence arrays
        pE = np.zeros_like(E_full)
        pE[non_zero_indx] = P_E
        write_list["pE.nrrd"] = pE
        
        uE = -1.0*np.ones_like(E_full)
        uE[non_zero_indx] = u_E_confidence
        write_list["UE.nrrd"] = uE

        #percent_uE = 1.0*np.ones_like(E_full)
        #percent_uE[non_zero_indx] = np.abs(div0(1.0, E_term_frac, value=-1.0) )
        #percent_uE[percent_uE > 1.0] = 1.0
        #write_list["percent_error_E.nrrd"] =  percent_uE
        
        pE1 = np.zeros_like(E1_full)
        pE1[non_zero_indx] = P_E1
        write_list["pE1.nrrd"] = pE1

        uE1 = -1.0*np.ones_like(E1_full)
        uE1[non_zero_indx] = u_E1_confidence
        write_list["UE1.nrrd"] = uE1

        #percent_uE1 = -1.0*np.ones_like(E1_full)
        #percent_uE1[non_zero_indx] = np.abs(div0(1.0,  E1_term_frac, value=-1.0))
        #percent_uE1[percent_uE1 > 1.0] = 1.0
        #write_list["percent_error_E1.nrrd"] = percent_uE1
    
        pE2 = np.zeros_like(E2_full)
        pE2[non_zero_indx] = P_E2
        write_list["pE2.nrrd"] = pE2
        
        uE2 = -1.0*np.ones_like(E2_full)
        uE2[non_zero_indx] = u_E2_confidence
        write_list["UE2.nrrd"] = uE2
        
        #percent_uE2 = -1.0*np.ones_like(E2_full)
        #percent_uE2[non_zero_indx] = np.abs(div0(1.0,  E2_term_frac, value=-1.0 ) )
        #percent_uE2[percent_uE2 > 1.0] = 1.0
        #write_list["percent_error_E2.nrrd"] =  percent_uE2

        pE3= np.zeros_like(E3_full)
        pE3[non_zero_indx] = P_E3
        write_list["pE1.nrrd"] = pE3

        uE3 = -1.0*np.ones_like(E3_full)
        uE3[non_zero_indx] = u_E3_confidence
        write_list["UE3.nrrd"] = uE3
        
        #percent_uE3 = -1.0*np.ones_like(E3_full)
        #percent_uE3[non_zero_indx] = np.abs(div0(1.0,  E3_term_frac, value=-1.0) )
        #percent_uE3[percent_uE3 > 1.0] = 1.0
        #write_list["percent_error_E3.nrrd"] = percent_uE3

        pE4 = np.zeros_like(E4_full)
        pE4[non_zero_indx] = P_E4
        write_list["pE4.nrrd"] = pE4
        
        uE4 = -1.0*np.ones_like(E4_full)
        uE4[non_zero_indx] = u_E4_confidence
        write_list["UE4.nrrd"] = uE4
        
        #percent_uE4 = -1.0*np.ones_like(E4_full)
        #percent_uE4[non_zero_indx] = np.abs(div0(1.0,  E4_term_frac, value=-1.0 ) )
        #percent_uE4[percent_uE4 > 1.0] = 1.0
        #write_list["percent_error_E4.nrrd"] = percent_uE4

        # threshold 
        #indx_E = np.where(E_full > uE)
        #E_thresh = np.zeros_like(E_full)
        #E_thresh[indx_E] = E_full[indx_E]
        #write_list["E_thresh.nrrd"] = E_thresh
        
        #indx_E1 = np.where(E1_full > uE1)
        #E1_thresh = np.zeros_like(E1_full)
        #E1_thresh[indx_E1] = E1_full[indx_E1]
        #write_list["E1_thresh.nrrd"] = E1_thresh

        #indx_E2 = np.where(E2_full > uE2)
        #E2_thresh = np.zeros_like(E2_full)
        #E2_thresh[indx_E2] = E2_full[indx_E2]
        #write_list["E2_thresh.nrrd"] = E2_thresh

        #indx_E3 = np.where(E3_full > uE3)
        #E3_thresh = np.zeros_like(E3_full)
        #E3_thresh[indx_E3] = E3_full[indx_E3]
        #write_list["E3_thresh.nrrd"] = E3_thresh
        
        #indx_E4 = np.where(E4_full > uE4)
        #E4_thresh = np.zeros_like(E4_full)
        #E4_thresh[indx_E4] = E4_full[indx_E4]
        #write_list["E4_thresh.nrrd"] = E4_thresh
        
        ## threshold 2
        #indx_E = np.where(E_full > 0.0)
        #E_thresh2= np.zeros_like(E_full)
        #E_thresh2[indx_E] = E_full[indx_E]
        #write_list["E_thresh2.nrrd"] = E_thresh2
        
        #indx_E1 = np.where(E1_full > 0.0)
        #E1_thresh2 = np.zeros_like(E1_full)
        #E1_thresh2[indx_E1] = E1_full[indx_E1]
        #write_list["E1_thresh2.nrrd"] = E1_thresh2

        #indx_E2 = np.where(E2_full > 0.0)
        #E2_thresh2 = np.zeros_like(E2_full)
        #E2_thresh2[indx_E2] = E2_full[indx_E2]
        #write_list["E2_thresh2.nrrd"] = E2_thresh2

        #indx_E3 = np.where(E3_full > 0.0)
        #E3_thresh2= np.zeros_like(E3_full)
        #E3_thresh2[indx_E3] = E3_full[indx_E3]
        #write_list["E3_thresh2.nrrd"] = E3_thresh2
        
        #indx_E4 = np.where(E4_full > 0.0)
        #E4_thresh2 = np.zeros_like(E4_full)
        #E4_thresh2[indx_E4] = E4_full[indx_E4]
        #write_list["E4_thresh2.nrrd"] = E4_thresh2


        for file_name, np_image in write_list.items():
            path_test = os.path.join(write_file_dir, case_id, write_dir, file_name)
            
            if ( not os.path.exists(path_test) or overwrite_out == True):
                nrrd.write(path_test, np_image, case_imgs["VWI_post_masked_vent"][1])

            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E.nrrd"), E_full, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E1.nrrd"), E1_full, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E2.nrrd"), E2_full, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E3.nrrd"), E3_full, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E4.nrrd"), E4_full, case_imgs["VWI_post_masked_vent"][1])
            
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE.nrrd"), uE, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE1.nrrd"), uE1, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE2.nrrd"), uE2, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE3.nrrd"), uE3, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE4.nrrd"), uE4, case_imgs["VWI_post_masked_vent"][1])
            
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E_thresh.nrrd"), E_thresh, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E1_thresh.nrrd"), E1_thresh, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E2_thresh.nrrd"), E2_thresh, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E3_thresh.nrrd"), E3_thresh, case_imgs["VWI_post_masked_vent"][1])
            #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E4_thresh.nrrd"), E4_thresh, case_imgs["VWI_post_masked_vent"][1])


    if (skip_bootstrap):
        pass
    else:
        boot_size = 10000
        #n_bins = 30
        pre_dist_std = np.zeros(boot_size)
        post_dist_std = np.zeros(boot_size)
        pre_dist_mean = np.zeros(boot_size)
        post_dist_mean = np.zeros(boot_size)

        for i in range(boot_size):
            X_resample_pre, ns = bootstrap_resample(img_pre_vent, n=None, percent=percent_boot)
            X_resample_post, ns = bootstrap_resample(img_post_vent, n=None, percent=percent_boot)
            pre_dist_std[i] =  X_resample_pre.std()
            post_dist_std[i] = X_resample_post.std()
            pre_dist_mean[i] =  X_resample_pre.mean()
            post_dist_mean[i] = X_resample_post.mean()
        #print( 'original mean:', X.mean()
        #print(case_id, "post vent resample MEAN: {0:.4f} pre vent resample MEAN {1:.4f}".format(
        #    X_resample_pre.mean() , X_resample_post.mean() ))
        
        print ("ha")
        gs = plt.GridSpec(3,2, wspace=0.2, hspace=0.8) 
        # Create a figure
        fig = plt.figure(figsize=(13, 13))
        fig.suptitle("{0} Comparison of bootstrapped means".format(case_id), fontsize = font_size)
        # SUBFIGURE 1
        # Create subfigure 1 (takes over two rows (0 to 1) and column 0 of the grid)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("{0}: pre vs post bootstrap".format(case_id), fontsize = font_size)

        ax3 = fig.add_subplot(gs[0, 1])
        ax3.set_title(r'$E_0 = \eta - \xi $ bootstrap', fontsize = font_size)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title(r'$E_1 = \frac{\overline{\xi}_{v}}{\overline{\eta}_{v}} post - pre$', fontsize = font_size)

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title(r'$E_2 = \frac{\eta}{\overline{\eta}_{v}} - \frac{\xi}{\overline{\xi}_{v}}$', fontsize = font_size)

        ax6 = fig.add_subplot(gs[2, 0])
        ax6.set_title(r'$E_3 = \frac{\eta - \overline{\eta}_{v}}{s_{\eta_{v}}} - \frac{\xi - \overline{\xi}_{v}}{s_{\xi_{v}}}$', fontsize = font_size)
        
        ax2 = fig.add_subplot(gs[2, 1])
        ax2.set_title("VWI_post_masked_vent bootstrap", fontsize = font_size)

        ax2.set_title(r'$ E_4 = \frac{ \left ( \eta - \overline{\eta}_{v} \right ) - \left ( \xi - \overline{\xi}_{v} \right ) } {\sqrt{s^2_{\eta_{v}} + s^2_{\eta_{v}}}} $', fontsize = font_size)


        ax1.scatter(pre_dist_mean, post_dist_mean)
        ax1.set_xlabel(r'$\eta$', fontsize = font_size)
        ax1.set_ylabel(r'$\xi$', fontsize = font_size)

        test_E = VWI_Enhancement(post_dist_mean, pre_dist_mean, 1.0, 1.0, kind = "E1")
        ax3.hist(test_E, bins="auto")
        ax3.axvline(x=(eta_vent - xi_vent), color='r')
        #ax3.axvline(x=test_E.mean(), color='b')
        ax3.set_xlabel("mean $E_0$", fontsize = font_size)
        ax3.set_ylabel("count", fontsize = font_size)

        test_E1 = VWI_Enhancement(post_dist_mean, pre_dist_mean, eta_vent, xi_vent, kind = "E1")
        ax4.hist(test_E1, bins="auto")
        ax4.axvline(x=0.0, color='r')
        #ax4.axvline(x=xi_vent/eta_vent*post_dist_mean.mean() - pre_dist_mean.mean(), color='r')
        #ax4.axvline(x=test_E1.mean(), color='b')
        ax4.set_xlabel("mean $E_1$", fontsize = font_size)
        ax4.set_ylabel("count", fontsize = font_size)

        test_E2 = VWI_Enhancement(post_dist_mean, pre_dist_mean, eta_vent, xi_vent, kind = "E2")
        ax5.hist(test_E2, bins="auto")
        ax5.axvline(x=0.0, color='r')
        #ax5.axvline(x=(post_dist_mean.mean() / eta_vent)  - (pre_dist_mean.mean() / xi_vent), color='r')
        #ax5.axvline(x=test_E2.mean(), color='b')
        ax5.set_xlabel("mean $E_2$", fontsize = font_size)
        ax5.set_ylabel("count", fontsize = font_size)

        test_E3 = VWI_Enhancement(post_dist_mean, pre_dist_mean, eta_vent, xi_vent, kind = "E3",
                        std_post_vent = post_dist_std, std_pre_vent = pre_dist_std)
        ax6.hist(test_E3, bins="auto")
        ax6.axvline(x=0.0 , color='r')
        #ax6.axvline(x=(post_dist_mean.mean() - eta_vent) /  post_dist_std.mean() - (pre_dist_mean.mean() - xi_vent) / pre_dist_std.mean() , color='b')
        ax6.set_xlabel("mean $E_3$", fontsize = font_size)
        ax6.set_ylabel("count", fontsize = font_size)

        test_E4 = VWI_Enhancement(post_dist_mean, pre_dist_mean, eta_vent, xi_vent, kind = "E4",
                        std_post_vent = post_dist_std, std_pre_vent = pre_dist_std)
        ax2.hist(test_E4, bins="auto")
        ax2.axvline(x=0.0 , color='r')
        #ax6.axvline(x=(post_dist_mean.mean() - eta_vent) /  post_dist_std.mean() - (pre_dist_mean.mean() - xi_vent) / pre_dist_std.mean() , color='b')
        ax2.set_xlabel("mean $E_4$", fontsize = font_size)
        ax2.set_ylabel("count", fontsize = font_size)

        path_bootstrap = os.path.join(write_file_dir, case_id, plots_dir, "Compare_bootstrap.png")
        bootstrap_fig_list.append(path_bootstrap)
        fig.savefig(path_bootstrap, dpi=dpi_value)
        #plt.show()
        plt.close(fig)
        del fig
        print("hehe")
        
    
#pickle.dump( params_list, open(os.path.join(write_file_dir, bw_file), 'wb') )

#print("saved another pickle!")

data_cases = {'Case ID': case_id_list, "Enhancement Type": e_type_list, "Region": label_list, "Average": average_list, "Uncertainty": uncertainty_list}
# Create DataFrame 
df_cases = pd.DataFrame(data_cases)

df_cases.to_pickle(os.path.join(write_file_dir, enhancement_file))

print("saved the pickle!")
print(image_path_list)
print(bootstrap_fig_list)


lg_8 = ax11_8.legend(bbox_to_anchor=(1.04,1), loc="upper left")
for lg in lg_8.legendHandles:
    lg.set_alpha(1.0)

path_ventricle_model = os.path.join(write_file_dir, "Compare_histograms.png")
fig8.savefig(path_ventricle_model, dpi=dpi_value)
plt.close(fig8)
del fig8


lg_5 = ax11_5.legend(bbox_to_anchor=(1.04,1), loc="upper left")
for lg in lg_5.legendHandles:
    lg.set_alpha(1.0)

path_ventricle_model = os.path.join(write_file_dir, "Compare_pre2post.png")
fig5.savefig(path_ventricle_model, dpi=dpi_value)
plt.close(fig5)
del fig5


handles, labels = ax11_5.get_legend_handles_labels()
lg_6 = ax11_6.legend(handles, labels, bbox_to_anchor=(1.04,1), loc="upper left")
#lg_6 = ax11_6.legend(bbox_to_anchor=(1.04,1), loc="upper left")
for lg in lg_6.legendHandles:
    lg.set_alpha(1.0)

path_ventricle_model = os.path.join(write_file_dir, "Compare_pre2post_ellipse.png")
fig6.savefig(path_ventricle_model, dpi=dpi_value)
plt.close(fig6)
del fig6
