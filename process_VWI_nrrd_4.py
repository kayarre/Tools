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

import matplotlib.pyplot as plt


def rician_eqn(p, measured_mean, measured_variance):
    A, sigma = p
    nu = A**2/ (4.0 * sigma**2.0)
    b =  (1+2.0*nu)*iv(0, nu) + 2.0*nu*iv(1,nu)
    mean = sigma *np.sqrt(np.pi/2.0)*np.exp(-nu)*(b) - measured_mean
    var = A + 2.0*sigma**2.0 - np.pi*sigma**2.0/2.0*np.exp(-2.0*nu)*b**2.0 - measured_variance
    return (mean, var)


def beta_N(N):
    """ return the Beta_N function
    @param: N the number of MRA channels
    """
    return np.sqrt(0.5*np.pi)*factorial2(2*N-1) / ( (2**(N-1)) * factorial(N-1) )
    
def xi(theta, N):
    """ returns the correction factor of the multi-channel MRI signal
    @param: theta the SNR of the guassian
    @param: N the number of MRA channels
    """
    return 2.0*N + theta**2 - (beta_N(N)**2.0*(hyp1f1(-0.5, N, -0.5*theta**2))**2)


def g_theta(theta, N, r):
    """ returns the guassian SNR value as a function of itself
    @param: theta the guassian SNR
    @param: N the number of MRA channels
    @param: r the measure signal to noise ratio
    """
    
    return np.sqrt(xi(theta, N)*(1.0 + r**2.0) - 2.0*N)

def koay_next(t_n, N, r):
    """ returns the n+1 guassian SNR value given an estimate
    @param: t_n estimate of the guassian SNR
    @param: N the number of MRA channels
    @param: r the measure signal to noise ratio
    """
    g_n = g_theta(t_n, N, r)
    b_n = beta_N(N)
    f1_a = hyp1f1(-0.5, N, -0.5*t_n**2.0)
    f1_b = hyp1f1(0.5, N+1, -0.5*t_n**2.0)
    return t_n - (g_n*(g_n - t_n) ) / (t_n*(1.0+r**2.0)*(1.0 - (0.5*b_n**2.0 / N) * f1_a * f1_b) - g_n)

def koay_test(M, s_r, theta):
    # doesn't work
    l_term = (1.0+0.5*theta**2.0)*iv(0, 0.25*theta**2.0) + 0.5*theta**2.0*iv(1, 0.25*theta**2.0)
    
    psi = 0.5*(np.sqrt(0.5*np.pi)*np.exp(-0.25*theta**2.0)*(l_term))
    s_g_m = M / psi
    
    xi_root = np.sqrt( theta**2.0 + 2.0 - 0.125*np.pi*np.exp(-0.5*theta**2.0)*(l_term**2.0))
    s_g_s =  s_r / xi_root 
    
    return s_g_m, s_g_s

def koay_test2(M, s_r, theta, N):
    l_term = hyp1f1(-0.5, N, -0.5*theta**2)
    beta_l_term = (beta_N(N)*l_term)
    
    s_g_m = M / (beta_l_term)
    
    xi_root = np.sqrt( 2.0*N + theta**2.0 - beta_N(N)**2.0*l_term**2.0 )
    s_g_s =  s_r / xi_root 
    
    #s_g_new  = max(s_g_s, s_g_m) # get the maximum deviation
    
    #M_g_new = s_g_new * beta_l_term
    
    return s_g_m, s_g_s

def koay_test3(M, s_r, theta, N):
    l_term = hyp1f1(-0.5, N, -0.5*theta**2)
    
    xi = 2.0*N + theta**2.0 - beta_N(N)**2.0*l_term**2.0 
    
    M_n_new = np.sqrt( M**2.0 + (1.0 - 2.0*N/xi)*s_r**2.0)
    
    s_g_new = M_n_new / theta
    #s_g_new  = max(s_g_s, s_g_m) # get the maximum deviation
    
    #M_g_new = s_g_new * beta_l_term
    
    return M_n_new, s_g_new

def lower_bound(N):
    """ return the lower bound of the estimation
    @param: N the number of MRA channels
    """
    return np.sqrt(2.0*N / xi(0.0, N) - 1.0)

def newton_koay(r, N, iterations=500, tolerance = 1.0E-9):
    """ returns  newton iteration solve to the Koay derived noise estimator
    @param: r the measured signal to noise ratio
    @param: N the number of MRA channels
    @param: iterations the maximum iterations for the newton solve
    @param tolerance the numerical tolerance of the newton iterations
    """
    #
    #https://www.sciencedirect.com/science/article/pii/S109078070600019X/
    it_default = np.copy(iterations)
    lb = lower_bound(N)
    if (np.isscalar(r)):
        if (r <= lb):
            t_1 = 0.0
            err = 0.0
        else:
            t_0 = r - lb # initial guess of the guassian SNR theta
            t_1 = koay_next(t_0, N, r)
            err = np.absolute( t_1 - t_0)
            while (err > tolerance and iterations >= 0):
                t_0 = np.copy(t_1)
                t_1  = koay_next(t_0, N, r)
                err = np.absolute( t_1 - t_0)
                iterations -= 1
            if (iterations < 0):
                print("{0}  iterations before reaching error tolerance, error: {1} tolerance:{2}".format(it_default, err, tolerance ))
    #else:
        #t_1 = np.empty(r.shape)
        #err = np.ones(r.shape)
        #indx = np.where(r <= lb)
        #t_1[indx] = 0.0
        #t_0 = r - lb
        #t_1[indx]  = koay_next(t_0[indx], N, r[indx])
        #while (err.any() > tolerance and iterations >= 0):
            #t_0 = np.copy(t_1)
            #t_1[indx]  = koay_next(t_0[indx], N, r[indx])
            #err = np.absolute( t_1 - t_0)
            #iterations -= 1
        #if (iterations < 0):
            #print("{0}  iterations before reaching error tolerance, error: {1} tolerance:{2}".format(it_default, err, tolerance ))
    
    return t_1, err

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
    elif kind == "E4":
        # ratio of normalized things, similar to E3
        E = ( std_pre_vent / std_post_vent ) * (post - mean_post_vent) / (pre - mean_pre_vent)
        return E
    elif kind == "E5":
        # ratio of normalized things, similar to E3
        num = np.sqrt(std_post_vent**2 + std_pre_vent**2)
        post_ = (post - mean_post_vent) / num
        pre_ = (pre - mean_pre_vent) / num 
    else:
        print("undefined enhancement kind")
        return 0.0
    
    E = post_ - pre_
    if return_parts:
        return E, post_, pre_
    else:
        return E

#def VWI_Uncertainty(kind = "E1", **kwargs):
    #""" calculate enhancement 
    #Parameters
    #----------
    #post : numpy array_like post contrast VWI
    #pre : numpy array_like pre contrast VWI
    #mean_post_vent : mean of post contrast ventricle
    #mean_pre_vent : mean of pre contrast ventricle
    #kind : which enhancement to calculate
    #std_post_vent : mean of post contrast ventricle
    #std_pre_vent : mean of pre contrast ventricle
    #-------
    #returns the enhancement calculation, numpy array_like
    #"""
    #if kind == "E1":
        ##"E = xi_vent / eta_vent * eta - xi"
        #E = (mean_pre_vent / mean_post_vent * post) - pre
    #elif kind == "E2":
        ##"E = eta / eta_vent - xi / xi_vent"
        #E = (post / mean_post_vent) - (pre / mean_pre_vent)
    #elif kind == "E3":
        ##"E = ( eta - mean_eta_vent) / stddev(eta_vent) - (xi - mean_xi_vent) / stddev(xi_vent)"
        #E = ( (post - mean_post_vent) / std_post_vent) - ( (pre - mean_pre_vent) / std_pre_vent )
    #elif kind == "E4":
        ## ratio of normalized things, similar to E3
        #E = ( std_pre_vent / std_post_vent ) * (post - mean_post_vent) / (pre - mean_pre_vent)
        
    #else:
        #print("undefined enhancement kind")
        #return 0.0
    #return E


def gumbel_r_fit(input_data, n_pts=1000, tail=0.05):
    # takes row or columnar format
    param = stats.gumbel_r.fit(input_data)
    arg = param[:-2]
    loc = param[-2]
    scale = param[-1]

    # Get sane start and end points of distribution
    start = stats.gumbel_r.ppf(tail, *arg, loc=loc, scale=scale) if arg else stats.gumbel_r.ppf(tail, loc=loc, scale=scale)
    end = stats.gumbel_r.ppf(1.0 - tail, *arg, loc=loc, scale=scale) if arg else stats.gumbel_r.ppf(1.0 - tail, loc=loc, scale=scale)
    x_test = np.linspace(start, end, n_pts)
    pdf_fitted = stats.gumbel_r.pdf(x_test, *arg, loc=loc, scale=scale)
    
    return pdf_fitted, x_test

#vars

conf = 2.0 #95% confidence interval
percent_boot = 0.01
font_size = 10

figure_1 = True
figure_2 = True


json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_normalization.json"

pickle_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_pickle.pkl"
write_file_dir = "/home/sansomk/caseFiles/mri/VWI_proj"
write_dir = "VWI_analysis"
plots_dir = "plots"
overwrite = 0
overwrite_out = False
skip_bootstrap = True

with open(json_file, 'r') as f:
    data = json.load(f)

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
      
    

#test = np.linspace(1,64,64, dtype=np.int)
#lb_test = lower_bound(test)
#plt.plot(test, lb_test)
#plt.show()

channels = int(1)
image_path_list = []
bootstrap_fig_list = []
for case_id, case_imgs  in image_dict.items():
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
    #zero_indx = np.where(case_imgs["post_float"][0] < 0.0)
    
    eta_model = case_imgs["model_post_masked"][0][case_imgs["model_post_masked"][0] >= 0.0]
    xi_model = case_imgs["model_pre2post_masked"][0][case_imgs["model_pre2post_masked"][0] >= 0.0]
    
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
    N_vent = float(img_post_vent.shape[0])
    cov_back = np.cov(np.vstack((img_post_back_inter, img_pre_back_inter)))
    #print(np.sqrt(2.0*np.trace(cov_vent) - np.sum(cov_vent)))

    eta_vent = np.mean(img_post_vent) # mean ventricle post
    xi_vent = np.mean(img_pre_vent) # mean ventricle pre
    u_eta_vent_2 = cov_vent[0,0] /  N_vent  # uncertainty vent post square
    u_xi_vent_2 = cov_vent[1,1] /  N_vent  # uncertainty vent pre square
    u_eta_xi_vent = cov_vent[0,1] / N_vent # covariance between pre and post
    
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
    
    E = VWI_Enhancement(eta, xi, 1.0, 1.0, kind = "E1")
    E1 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E1")
    E2 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E2")
    E3 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2))
    #E4 = VWI_Enhancement(eta, xi, eta_vent, xi_vent, kind = "E4",
                                                        #std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        #std_pre_vent = np.sqrt(u_xi_vent_2))
    
    #E4_param = stats.cauchy.fit(E4)
    #print("cauchy model params", E4_param)
    ## Separate parts of parameters
    #arg = E4_param[:-2]
    #loc = E4_param[-2]
    #scale = E4_param[-1]

    ## Get sane start and end points of distribution
    #start = stats.cauchy.ppf(0.01, *arg, loc=loc, scale=scale) if arg else stats.cauchy.ppf(0.05, loc=loc, scale=scale)
    #end = stats.cauchy.ppf(0.99, *arg, loc=loc, scale=scale) if arg else stats.cauchy.ppf(0.95, loc=loc, scale=scale)
    #E4_x_test = np.linspace(start, end, 30000)
    #E4_pdf_fitted = stats.cauchy.pdf(E4_x_test, *arg, loc=loc, scale=scale)
    
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

    E5_model, E5_post_model, E5_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E5",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)

    E_PI, E_post_PI, E_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           1.0, 1.0, kind = "E1", return_parts=True)


    gs4 = plt.GridSpec(2,2, wspace=0.2, hspace=0.8)
    fig4 = plt.figure(figsize=(13, 13))
    ax1_4 = fig4.add_subplot(gs4[0, 0])
    ax1_4.set_title("Compare E pre vs post", fontsize = font_size)
    ax1_4.scatter(E_pre_model.ravel(), E_post_model.ravel())
    ax1_4.set_xlabel(r'pre $\xi$', fontsize = font_size)
    ax1_4.set_ylabel(r'post $\eta$', fontsize = font_size)
    
    ax2_4 = fig4.add_subplot(gs4[0, 1])
    ax2_4.set_title("Compare E1 pre vs post", fontsize = font_size)
    ax2_4.scatter(E1_pre_model.ravel(), E1_post_model.ravel())
    ax2_4.set_xlabel(r'pre $\xi$', fontsize = font_size)
    ax2_4.set_ylabel(r'post $\frac{ \bar{\xi}_{vent}}{ \bar{\eta}_{vent}} \eta$', fontsize = font_size)
    
    ax3_4 = fig4.add_subplot(gs4[1, 0])
    ax3_4.set_title("Compare E2 pre vs post", fontsize = font_size)
    ax3_4.scatter(E2_pre_model.ravel(), E2_post_model.ravel())
    ax3_4.set_xlabel(r'pre $\frac{\xi}{\bar{\xi}_{vent}}$', fontsize = font_size)
    ax3_4.set_ylabel(r'post $\frac{\eta}{\bar{\eta}_{vent}}$', fontsize = font_size)
    
    ax4_4 = fig4.add_subplot(gs4[1, 1])
    ax4_4.set_title("Compare E3 pre vs post", fontsize = font_size)
    ax4_4.scatter(E3_pre_model.ravel(), E3_post_model.ravel())
    ax4_4.set_xlabel(r'pre $ \frac{\xi - \bar{\xi}_{vent}}{ \sigma_{ \xi_{vent} } }$', fontsize = font_size)
    ax4_4.set_ylabel(r'post $\frac{\eta - \bar{\eta}_{vent}}{\sigma_{ \eta_{vent} } }$', fontsize = font_size)
    
    path_E3_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_E3_model.png")
    fig4.savefig(path_E3_model)
    del fig4
    
    gs5 = plt.GridSpec(2,2, wspace=0.2, hspace=0.8)
    fig5 = plt.figure(figsize=(13, 13))
    ax1_5 = fig5.add_subplot(gs5[:, 0])
    ax1_5.set_title("Compare ventricle pre vs post", fontsize = font_size)
    ax1_5.scatter(img_pre_vent, img_post_vent)
    ax1_5.axis('equal')
    ax1_5.set_xlabel(r'pre $\xi_{ventricle}$', fontsize = font_size)
    ax1_5.set_ylabel(r'post $\eta_{ventricle}$', fontsize = font_size)
    
    ax2_5 = fig5.add_subplot(gs5[:, 1])
    ax2_5.set_title("Compare Pituitary Infindibulum pre vs post", fontsize = font_size)
    ax2_5.scatter(xi_PI, eta_PI)
    ax2_5.axis('equal')
    ax2_5.set_xlabel(r'pre $\xi_{ventricle}$', fontsize = font_size)
    ax2_5.set_ylabel(r'post $\eta_{ventricle}$', fontsize = font_size)
    
    path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_PI_Ventricle_data.png")
    fig5.savefig(path_ventricle_model)
    del fig5
    
    
    gs6 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    fig6 = plt.figure(figsize=(13, 13))
    ax1_6 = fig6.add_subplot(gs5[0, 0])
    ax1_6.set_title("Compare E5 pre vs post", fontsize = font_size)
    ax1_6.scatter(E5_pre_model, E5_post_model)
    ax1_6.axis('equal')
    ax1_6.set_xlabel(r'post $\frac{\xi- \bar{\xi}_{vent}}{\sqrt{ \sigma^2_{ \eta_{vent} } + \sigma^2_{ \xi_{vent} } } }$', fontsize = font_size)
    ax1_6.set_ylabel(r'post $\frac{\eta - \bar{\eta}_{vent}}{\sqrt{ \sigma^2_{ \eta_{vent} } + \sigma^2_{ \xi_{vent} } } }$', fontsize = font_size)
    
    path_ventricle_model = os.path.join(write_file_dir, case_id, plots_dir, "Compare_E5_data.png")
    fig6.savefig(path_ventricle_model)
    del fig6
    
    
    
    #E4_model = VWI_Enhancement(eta_model, xi_model, eta_vent, xi_vent, kind = "E4",
                                                        #std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        #std_pre_vent = np.sqrt(u_xi_vent_2))
    
    #E4_model_param = stats.cauchy.fit(E4_model)
    ## Separate parts of parameters
    #arg = E4_model_param[:-2]
    #loc = E4_param[-2]
    #scale = E4_model_param[-1]
    #print("cauchy model params", E4_model_param)

    ## Get sane start and end points of distribution
    #start = stats.cauchy.ppf(0.05, *arg, loc=loc, scale=scale) if arg else stats.cauchy.ppf(0.05, loc=loc, scale=scale)
    #end = stats.cauchy.ppf(0.95, *arg, loc=loc, scale=scale) if arg else stats.cauchy.ppf(0.95, loc=loc, scale=scale)
    #E4_model_x_test = np.linspace(start, end, 10000)
    #E4_model_pdf_fitted = stats.cauchy.pdf(E4_model_x_test, *arg, loc=loc, scale=scale)

    
    # determine which term is the driver for uncertainty
    
    u_E_2 = (u_eta_back_2 + u_xi_back_2 - 2.0 * u_eta_xi_back )
    u_E = np.sqrt(u_E_2)
    u_E_confidence = 2.0 * u_E # gaussian 95% confidence interval
    
    u_E1_2 =  eta_vent_term + xi_vent_term + eta_term + xi_term + eta_xi_term + eta_xi_vent_term
    u_E1 = np.sqrt(u_E1_2)
    u_E1_confidence = 2.0 * u_E1 # gaussian 95% confidence interval

    
    u_E2_2 = (1.0 / (eta_vent**2.0) * u_eta_back_2 +
                    1.0 / (xi_vent**2.0) * u_xi_back_2 + 
                    np.square( eta / (eta_vent**2.0)) * u_eta_vent_2 +
                    np.square( xi / (xi_vent**2.0)) * u_xi_vent_2 -
                    2.0 / (eta_vent * xi_vent) * u_eta_xi_back -
                    2.0 * (eta * xi) / ( (eta_vent * xi_vent)**2.0 ) * u_eta_xi_vent
                    )
    u_E2 = np.sqrt(u_E2_2)
    u_E2_confidence = 2.0 * u_E2 # gaussian 95% confidence interval
    
    u_E3_2 = (1.0 / (u_eta_vent_2) * u_eta_back_2 +
                    1.0 / (u_xi_vent_2) * u_xi_back_2 +
                    2.0 / N_vent -
                    2.0 / (np.sqrt(u_eta_vent_2 *u_xi_vent_2)) * u_eta_xi_back -
                    2.0 / (np.sqrt(u_eta_vent_2 *u_xi_vent_2)) * u_eta_xi_vent +
                    np.square(eta - eta_vent) / u_eta_vent_2  * (N_vent / (N_vent - 1.0)) +
                    np.square(xi - xi_vent) / u_xi_vent_2  * (N_vent / (N_vent - 1.0))
                    )
    u_E3 = np.sqrt(u_E3_2)
    u_E3_confidence = 2.0 * u_E3 # gaussian 95% confidence interval
    

    
    print(np.mean(u_E_2), np.mean(u_E), np.mean(u_E_confidence))
    print(np.mean(u_E1_2), np.mean(u_E1), np.mean(u_E1_confidence))
    print(np.mean(u_E2_2), np.mean(u_E2), np.mean(u_E2_confidence))
    print("pink")
    #u_E_percent = 100.0*np.divide(u_E_confidence, np.absolute(E), out=-1.0*np.ones_like(E), where=E!=0)
    #u_E1_percent = 100.0*np.divide(u_E1_confidence, np.absolute(E1), out=-1.0*np.ones_like(E1), where=E1!=0)
    #u_E2_percent = 100.0*np.divide(u_E2_confidence, np.absolute(E2), out=-1.0*np.ones_like(E2), where=E2!=0)

    #print(np.mean(u_E_percent), np.mean(u_E1_percent), np.mean(u_E2_percent))
    #print(np.std(u_E_percent), np.std(u_E1_percent), np.std(u_E2_percent))
    #print(np.mean(eta_vent_term ),  np.mean(xi_vent_term ), eta_term, xi_term, np.mean(eta_xi_term), np.mean(eta_xi_vent_term))

    n_bins2 = 4000
    n_bins3 = 100
    gs2 = plt.GridSpec(4,4, wspace=0.2, hspace=0.8)
    #gs2 = plt.GridSpec(4,4, wspace=0.2, hspace=0.8) 
    # Create a figure
    fig2 = plt.figure(figsize=(13, 13))

    # SUBFIGURE 1
    # Create subfigure 1 (takes over two rows (0 to 1) and column 0 of the grid)
    ax1_2 = fig2.add_subplot(gs2[0, 0:2])
    ax1_2.set_title("{0}: Enhancement  Volume".format(case_id), fontsize = font_size)
    ax2_2 = fig2.add_subplot(gs2[1, 0:2])
    ax2_2.set_title("Enhancement 1 Volume", fontsize = font_size)
    
    ax3_2 = fig2.add_subplot(gs2[2,0:2])
    ax3_2.set_title("Enhancement 2 Volume", fontsize = font_size)
    #ax4_2 = fig2.add_subplot(gs2[3, 0])
    #ax4_2.set_title(r'Enhancement 1 uncertainty', fontsize = font_size)
    ax9_2 = fig2.add_subplot(gs2[3, 0:2])
    ax9_2.set_title("Enhancement 3 Volume", fontsize = font_size)
    
    ax5_2 = fig2.add_subplot(gs2[0, 2:])
    ax5_2.set_title("{0}: Enhancement model".format(case_id), fontsize = font_size)
    ax6_2 = fig2.add_subplot(gs2[1, 2:])
    ax6_2.set_title("Enhancement 1 model", fontsize = font_size)
    
    ax7_2 = fig2.add_subplot(gs2[2, 2:])
    ax7_2.set_title("Enhancement 2 model", fontsize = font_size)
    #ax8_2 = fig2.add_subplot(gs2[3, 2:])
    #ax8_2.set_title(r'Enhancement 1 uncertainty', fontsize = font_size)
    ax10_2 = fig2.add_subplot(gs2[3, 2:])
    ax10_2.set_title("Enhancement 3 model", fontsize = font_size)
    

    #E_pdf_fitted, E_x_test = gumbel_r_fit(E.ravel(), n_pts=2000, tail=0.05)
    #E1_pdf_fitted, E1_x_test = gumbel_r_fit(E1.ravel(), n_pts=2000, tail=0.05)
    #E2_pdf_fitted, E2_x_test = gumbel_r_fit(E2.ravel(), n_pts=2000, tail=0.05)
    #E3_pdf_fitted, E3_x_test = gumbel_r_fit(E3.ravel(), n_pts=2000, tail=0.05)

    ax1_2.hist(E.ravel(), bins=n_bins2, label="hist")
    ax1_2.axvline(x=np.mean(E), color='r')
    #ax1_2.plot(E_x_test, E_pdf_fitted, label="pdf")
    
    ax2_2.hist(E1.ravel(), bins=n_bins2)
    ax2_2.axvline(x=np.mean(E1), color='r')
    #ax2_2.plot(E1_x_test, E1_pdf_fitted, label="pdf")
    
    ax2_2.set_xlabel("Enhancement", fontsize = font_size)
    ax1_2.set_ylabel("count", fontsize = font_size)
    ax2_2.set_ylabel("count", fontsize = font_size)
    
    ax3_2.hist(E2.ravel(), bins=n_bins2)
    ax3_2.axvline(x=np.mean(E2), color='r')
    #ax3_2.plot(E2_x_test, E2_pdf_fitted, label="pdf")
    ax3_2.set_xlabel("Enhancement", fontsize = font_size)
    ax3_2.set_ylabel("count", fontsize = font_size)
    
    ax9_2.hist( E3, bins=n_bins2)
    ax9_2.axvline(x=np.mean(E3), color='r')
    #ax9_2.plot(E3_x_test, E3_pdf_fitted, label="pdf")
    #ax9_2.plot(E4_pdf_fitted)
    ax9_2.set_xlabel("Enhancement", fontsize = font_size)
    ax9_2.set_ylabel("count", fontsize = font_size)
    
    #ax4_2.hist(u_E_confidence.ravel(), bins=n_bins2)
    #ax4_2.axvline(x=np.mean(u_E_confidence), color='r')gs2 = plt.GridSpec(4,4, wspace=0.2, hspace=0.8)
    #ax4_2.set_ylabel("count", fontsize = font_size)


    #E_pdf_fitted_model, E_x_model = gumbel_r_fit(E.ravel(), n_pts=2000, tail=0.05)
    #E1_pdf_fitted_model, E1_x_model = gumbel_r_fit(E1.ravel(), n_pts=2000, tail=0.05)
    #E2_pdf_fitted_model, E2_x_model = gumbel_r_fit(E2.ravel(), n_pts=2000, tail=0.05)
    #E3_pdf_fitted_model, E3_x_model = gumbel_r_fit(E3.ravel(), n_pts=2000, tail=0.05)

    ax5_2.hist(E_model.ravel(), bins=n_bins3)
    ax5_2.axvline(x=np.mean(E_model), color='r')
    #ax5_2.plot(E_x_model, E_pdf_fitted_model, label="pdf")
    ax5_2.set_ylabel("count", fontsize = font_size)
    
    ax6_2.hist(E1_model.ravel(), bins=n_bins3)
    ax6_2.set_ylabel("count", fontsize = font_size)
    ax6_2.axvline(x=np.mean(E1_model), color='r')
    #ax6_2.plot(E1_x_model, E1_pdf_fitted_model, label="pdf")
    ax6_2.set_xlabel("Enhancement", fontsize = font_size)
    
    ax7_2.hist(E2_model.ravel(), bins=n_bins3)
    ax7_2.axvline(x=np.mean(E2_model), color='r')
    #ax7_2.plot(E2_x_model, E2_pdf_fitted_model, label="pdf")
    ax7_2.set_xlabel("Enhancement", fontsize = font_size)
    ax7_2.set_ylabel("count", fontsize = font_size)

    ax10_2.hist( E3_model, bins=n_bins3)
    ax10_2.axvline(x=np.mean(E3_model), color='r')
    #ax10_2.plot(E3_x_model, E3_pdf_fitted_model, label="pdf")
    #ax10_2.plot(E_model_pdf_fitted)
    ax10_2.set_xlabel("Enhancement", fontsize = font_size)
    ax10_2.set_ylabel("count", fontsize = font_size)
    
    #ax8_2.hist(u_E_confidence.ravel(), bins=n_bins3)
    #ax8_2.axvline(x=np.mean(u_E_confidence), color='r')
    #ax8_2.set_ylabel("count", fontsize = font_size)
    path_images = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Enhancement_Distribution.png")
    
    try:
        # Create target Directory
        test = os.path.join(write_file_dir, case_id, plots_dir)
        os.mkdir(test)
        print("Directory " , test ,  " Created ") 
    except FileExistsError:
        print("Directory " , test ,  " already exists")
    
    image_path_list.append(path_images)
    fig2.savefig(path_images)
    del fig2

    #plt.show()

    write_list = {}
    E_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], 1.0, 1.0, kind = "E1")
    write_list["E.nrrd"] = E_full
    E1_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E1")
    write_list["E1.nrrd"] = E1_full
    E2_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E2")
    write_list["E2.nrrd"] = E2_full
    E3_full, E3_post, E3_pre = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2), 
                                                        return_parts=True)
    write_list["E3.nrrd"] = E3_full
    E5_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E5",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2), 
                                                        return_parts=False)
    write_list["E5.nrrd"] = E5_full

    gs3 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    fig3 = plt.figure(figsize=(13, 13))
    ax1_3 = fig3.add_subplot(gs3[0, 0])
    ax1_3.set_title("Compare E3 pre vs post", fontsize = font_size)
    ax1_3.scatter(E3_pre.ravel(), E3_post.ravel())
    ax1_3.set_xlabel(r'pre $ \frac{\xi - \bar{\xi}_{vent}}{std(\xi_{vent})}$', fontsize = font_size)
    ax1_3.set_ylabel(r'post $\frac{\eta - \bar{\eta}_{vent}}{std(\eta_{vent})}$', fontsize = font_size)
    
    path_E3 = os.path.join(write_file_dir, case_id, plots_dir, "Compare_E3.png")
    fig3.savefig(path_E3)
    del fig3

    #uE = -999.0*np.ones_like(E_full)
    #uE[non_zero_indx] = u_E_percent
    #uE1 = -999.0*np.ones_like(E1_full)
    #uE1[non_zero_indx] = u_E1_percent
    #uE2 = -999.0*np.ones_like(E2_full)
    #uE2[non_zero_indx] = u_E2_percent
    
    # create confidence arrays
    uE = -1.0*np.ones_like(E_full)
    uE[non_zero_indx] = u_E_confidence
    write_list["UE.nrrd"] = uE
    uE1 = -1.0*np.ones_like(E1_full)
    uE1[non_zero_indx] = u_E1_confidence
    write_list["UE1.nrrd"] = uE1
    uE2 = -1.0*np.ones_like(E2_full)
    uE2[non_zero_indx] = u_E2_confidence
    write_list["UE2.nrrd"] = uE2
    uE3 = -1.0*np.ones_like(E3_full)
    uE3[non_zero_indx] = u_E3_confidence
    write_list["UE3.nrrd"] = uE3

    # threshold 
    indx_E = np.where(E_full > uE)
    E_thresh = np.zeros_like(E_full)
    E_thresh[indx_E] = E_full[indx_E]
    write_list["E_thresh.nrrd"] = E_thresh
    
    indx_E1 = np.where(E1_full > uE1)
    E1_thresh = np.zeros_like(E1_full)
    E1_thresh[indx_E1] = E1_full[indx_E1]
    write_list["E1_thresh.nrrd"] = E1_thresh

    indx_E2 = np.where(E2_full > uE2)
    E2_thresh = np.zeros_like(E2_full)
    E2_thresh[indx_E2] = E2_full[indx_E2]
    write_list["E2_thresh.nrrd"] = E2_thresh

    indx_E3 = np.where(E3_full > uE3)
    E3_thresh = np.zeros_like(E3_full)
    E3_thresh[indx_E3] = E3_full[indx_E3]
    write_list["E3_thresh.nrrd"] = E3_thresh
    
    
    for file_name, np_image in write_list.items():
        path_test = os.path.join(write_file_dir, case_id, write_dir, file_name)
        
        if ( not os.path.exists(path_test) or overwrite_out == True):
            nrrd.write(path_test, np_image, case_imgs["VWI_post_masked_vent"][1])
        #E4_full = VWI_Enhancement(case_imgs["post_float"][0], case_imgs["pre2post"][0], eta_vent, xi_vent, kind = "E4",
                                                            #std_post_vent = np.sqrt(u_eta_vent_2), 
                                                            #std_pre_vent = np.sqrt(u_xi_vent_2))
        #nrrd.write("test_interval_2.nrrd", u_E_2, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write("test_interval.nrrd", u_E_confidence, case_imgs["VWI_post_masked_vent"][1])
        
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E1.nrrd"), E1_full, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E2.nrrd"), E2_full, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E3.nrrd"), E3_full, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E5.nrrd"), E5_full, case_imgs["VWI_post_masked_vent"][1])
        
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE.nrrd"), uE, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE1.nrrd"), uE1, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE2.nrrd"), uE2, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "UE3.nrrd"), uE3, case_imgs["VWI_post_masked_vent"][1])
        
        ##nrrd.write(os.path.join(write_file_dir, write_dir, "E4.nrrd"), E4_full, case_imgs["VWI_post_masked_vent"][1])
        
        
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E_thresh.nrrd"), E_thresh, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E1_thresh.nrrd"), E1_thresh, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E2_thresh.nrrd"), E2_thresh, case_imgs["VWI_post_masked_vent"][1])
        #nrrd.write(os.path.join(write_file_dir, case_id, write_dir, "E3_thresh.nrrd"), E3_thresh, case_imgs["VWI_post_masked_vent"][1])


    if (skip_bootstrap):
        pass
    else:
        boot_size = 10000
        n_bins = 30
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

        # SUBFIGURE 1
        # Create subfigure 1 (takes over two rows (0 to 1) and column 0 of the grid)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("{0}: VWI_pre_masked_vent bootstrap".format(case_id), fontsize = font_size)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("VWI_post_masked_vent bootstrap", fontsize = font_size)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title(r'$E = post-pre$ bootstrap', fontsize = font_size)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title(r'$E_1 = \frac{\overline{pre}_{vent}}{\overline{post}_{vent}} post - pre$', fontsize = font_size)

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_title(r'$E_2 = \frac{post}{\overline{post}_{vent}} - \frac{pre}{\overline{pre}_{vent}}$', fontsize = font_size)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_title(r'$E_3 = \frac{post - \overline{post}_{vent}}{s_{post_{vent}}} - \frac{pre - \overline{pre}_{vent}}{s_{pre_{vent}}}$', fontsize = font_size)


        ax1.hist(pre_dist_mean, bins=n_bins)
        ax1.axvline(x=xi_vent, color='r')
        ax1.set_xlabel("mean", fontsize = font_size)
        ax1.set_ylabel("count", fontsize = font_size)
        
        ax2.hist(post_dist_mean, bins=n_bins)
        ax2.axvline(x=eta_vent, color='r')
        ax2.set_xlabel("mean ", fontsize = font_size)
        ax2.set_ylabel("count", fontsize = font_size)

        test_E = VWI_Enhancement(post_dist_mean, pre_dist_mean, 1.0, 1.0, kind = "E1")
        ax3.hist(test_E, bins=n_bins)
        ax3.axvline(x=(eta_vent - xi_vent), color='r')
        #ax3.axvline(x=test_E.mean(), color='b')
        ax3.set_xlabel("mean E", fontsize = font_size)
        ax3.set_ylabel("count", fontsize = font_size)

        test_E1 = VWI_Enhancement(post_dist_mean, pre_dist_mean, eta_vent, xi_vent, kind = "E1")
        ax4.hist(test_E1, bins=n_bins)
        ax4.axvline(x=0.0, color='r')
        #ax4.axvline(x=xi_vent/eta_vent*post_dist_mean.mean() - pre_dist_mean.mean(), color='r')
        #ax4.axvline(x=test_E1.mean(), color='b')
        ax4.set_xlabel("mean E1", fontsize = font_size)
        ax4.set_ylabel("count", fontsize = font_size)

        test_E2 = VWI_Enhancement(post_dist_mean, pre_dist_mean, eta_vent, xi_vent, kind = "E2")
        ax5.hist(test_E2, bins=n_bins)
        ax5.axvline(x=0.0, color='r')
        #ax5.axvline(x=(post_dist_mean.mean() / eta_vent)  - (pre_dist_mean.mean() / xi_vent), color='r')
        #ax5.axvline(x=test_E2.mean(), color='b')
        ax5.set_xlabel("mean E2", fontsize = font_size)
        ax5.set_ylabel("count", fontsize = font_size)

        test_E3 = VWI_Enhancement(post_dist_mean, pre_dist_mean, eta_vent, xi_vent, kind = "E3",
                        std_post_vent = post_dist_std, std_pre_vent = pre_dist_std)
        ax6.hist(test_E3, bins=n_bins)
        ax6.axvline(x=0.0 , color='r')
        #ax6.axvline(x=(post_dist_mean.mean() - eta_vent) /  post_dist_std.mean() - (pre_dist_mean.mean() - xi_vent) / pre_dist_std.mean() , color='b')
        ax6.set_xlabel("mean E3", fontsize = font_size)
        ax6.set_ylabel("count", fontsize = font_size)

        path_bootstrap = os.path.join(write_file_dir, case_id, plots_dir, "Compare_bootstrap.png")
        bootstrap_fig_list.append(path_bootstrap)
        fig.savefig(path_bootstrap)
        #plt.show()
        del fig
        print("hehe")
        
print("hooray")
print(image_path_list)
print(bootstrap_fig_list)

    

