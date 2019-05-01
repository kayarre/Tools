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
    X_resample = np.random.choice(X,p_n)
    return X_resample, p_n



#vars

conf = 2.0 #95% confidence interval
percent_boot = 0.01
font_size = 10


json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_normalization.json"

pickle_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_pickle.pkl"
overwrite = 0

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
                ("VWI_background_post_intersection", "VWI_background_pre_intersection")
                ]
image_dict = {}



#case_id_list = []
#for case_id, images in data.items():5
    #case_id_list.append(case_id)

#for image_label in subset_labels:
    #image_dict[image_label] = {}
    #print(image_label)
    #for case_id in case_id_list:
        #mask_path = data[case_id][image_label]   
        ##vwi_mask, vwi_mask_header = nrrd.read(vwi_mask_path)
        #image_tuple = nrrd.read(mask_path)
        #image_dict[image_label][case_id] = image_tuple
        #vwi_mask_array = image_tuple[0][image_tuple[0] >= 0.0]
        ##print(image_tuple[0].size - vwi_mask_array.shape[0])
        #print(case_id, "mean: {0}".format(np.mean(vwi_mask_array)),
            #" std: {0}".format(np.std(vwi_mask_array)),
            #" sample size {0}".format(vwi_mask_array.shape[0]))
    ##print(np.std(vwi_mask_array), np.std(pre2post_mask_array))

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
            #vwi_pre = image_tuple_pre[0][image_tuple_pre[0] >= 0.0]
            #mean_pre = np.mean(vwi_pre)
            #std_pre = np.std(vwi_pre)
            
            #vwi_post = image_tuple_post[0][image_tuple_post[0] >= 0.0]
            #mean_post = np.mean(vwi_post)
            #std_post = np.std(vwi_post)
            
            #U_pre = conf * std_pre
            #U_post = conf * std_post
            
            #SNR_post = mean_post/std_post
            #SNR_pre = mean_pre/std_pre
            
            #U_E = np.sqrt(U_post**2 + U_pre**2)
            
            #M_bar_pre = std_pre / np.sqrt5(2.0-np.pi/2.0) * np.sqrt(np.pi/2.0)
            
            #M_bar_post = std_post / np.sqrt(2.0-np.pi/2.0) * np.sqrt(np.pi/2.0)
            
            ##E = image_tuple_post[0] - image_tuple_pre[0]
            #e_str = ""
            #if ( post_label != "VWI_background_post_masked"):
                #e_str = "U_E: {0:.4f}".format(U_E)
            #print(post_label)
            #print( "mean post: {0:.4f} mean pre: {1:.4f}".format(mean_post, mean_pre),
                    #"std post: {0:.4f} std pre: {1:.4f}".format(std_post, std_pre),
                    #"U_post: {0:.4f} U_pre: {1:.4f} {2}".format(U_post, U_pre, e_str),
                    #"M_bar_post: {0:.4f} M_bar_pre: {1:.4f}".format(M_bar_post, M_bar_pre)
                    #)
            #print( "SNR_post: {0:.4f} SNR_pre: {1:.4f}".format(SNR_post, SNR_pre)
                    #)

            #" std: {0}".format(np.std(vwi_mask_array)),
            #" sample size {0}".format(vwi_mask_array.shape[0]))
            
            #print(image_label, "mean: {0}".format(np.mean(vwi_mask_array)),
                #" std: {0}".format(np.std(vwi_mask_array)),
                #" sample size {0}".format(vwi_mask_array.shape[0]))
        #print(np.std(vwi_mask_array), np.std(pre2post_mask_array))
    pickle.dump(image_dict,  open(pickle_file, "wb"))

else:
    with open(pickle_file, "rb") as pkl_f:
        image_dict = pickle.load(pkl_f)
        pickle_time = os.path.getmtime(pickle_file)
      
    dump = False
    for case_id, images in data.items():
        print(case_id)
        for post_label, pre_label  in groups:
            #print(pre_label, post_label)
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
for case_id, case_imgs  in image_dict.items():
    img_post_vent = case_imgs["VWI_post_masked_vent"][0][case_imgs["VWI_post_masked_vent"][0] >= 0.0]
    img_post_back = case_imgs["VWI_background_post_masked"][0][case_imgs["VWI_background_post_masked"][0] >= 0.0]
    img_post_model = case_imgs["model_post_masked"][0][case_imgs["model_post_masked"][0] >= 0.0]
    img_post_back_inter = case_imgs["VWI_background_post_intersection"][0][case_imgs["VWI_background_post_intersection"][0] >= 0.0]
    
    img_pre_vent = case_imgs["VWI_pre2post_masked_vent"][0][case_imgs["VWI_pre2post_masked_vent"][0] >= 0.0]
    img_pre_back = case_imgs["VWI_background_pre_masked"][0][case_imgs["VWI_background_pre_masked"][0] >= 0.0]
    img_pre_model = case_imgs["model_pre2post_masked"][0][case_imgs["model_pre2post_masked"][0] >= 0.0]
    img_pre_back_inter = case_imgs["VWI_background_pre_intersection"][0][case_imgs["VWI_background_pre_intersection"][0] >= 0.0]
    
    scale_factor = np.sqrt(2.0-np.pi/2.0) 
    #post_back_noise = np.mean(img_post_back) / np.sqrt(np.pi/2.0) # rayleigh distribution
    #pre_back_noise = np.mean(img_pre_back) / np.sqrt(np.pi/2.0) # rayleigh distribution
    
    back_std_pre_scale = np.std(img_pre_back) #/ scale_factor
    back_std_post_scale = np.std(img_post_back) #/ scale_factor 
    
    SNR_post_vent  = np.mean(img_post_vent) /  back_std_post_scale #post_back_noise
    SNR_pre_vent  = np.mean(img_pre_vent) / back_std_pre_scale #pre_back_noise
    #print(case_id, "post vent size: {0:} pre vent size {1:}".format(SNR_post_vent.shape, SNR_pre_vent.shape))
    #print(case_id, "post vent SNR: {0:.4f} pre vent SNR {1:.4f}".format(SNR_post_vent, SNR_pre_vent))
    print(case_id, "post vent MEAN: {0:.4f} pre vent MEAN {1:.4f}".format(np.mean(img_post_vent), np.mean(img_pre_vent)))
    print(case_id, "post vent STD: {0:.4f} pre vent STD {1:.4f}".format(np.std(img_post_vent) , np.std(img_pre_vent) ))
    
    print(case_id, "post back MEAN: {0:.4f} pre back MEAN {1:.4f}".format(np.mean(img_post_back) , np.mean(img_pre_back) ))
    print(case_id, "post inter MEAN: {0:.4f} pre inter MEAN {1:.4f}".format(np.mean(img_post_back_inter) , np.mean(img_pre_back_inter) ))
    #print(case_id, "post vent inter shape: {0} pre vent inter shape {1}".format(img_post_back_inter.shape ,img_pre_back_inter.shape ))
    
    print(case_id, "post back STD: {0:.4f} pre back STD {1:.4f}".format(back_std_post_scale, back_std_pre_scale))
    print(case_id, "post inter STD: {0:.4f} pre inter STD {1:.4f}".format(np.std(img_post_back_inter) , np.std(img_pre_back_inter) ))    
    #koay_result_post, err = newton_koay(SNR_post_vent, channels)
    
    
    #print("SNR : {0} ".format(SNR_post_vent))
    #print(koay_result_post, err)
    ##print(koay_test(np.mean(img_post_vent), back_std_post_scale , SNR_post_vent))
    #print(koay_test2(np.mean(img_post_vent), back_std_post_scale , koay_result_post, channels))
    #M_new_post, s_new_post = koay_test3(np.mean(img_post_vent), back_std_post_scale , koay_result_post, channels)
    
    #print(M_new_post, s_new_post)
    
    #koay_result_pre, err = newton_koay(SNR_pre_vent, channels)

    #print("SNR : {0} ".format(SNR_pre_vent))
    #print(koay_result_pre, err)
    ##print(koay_test(np.mean(img_pre_vent), back_std_pre_scale , SNR_post_vent))
    #print(koay_test2(np.mean(img_pre_vent), back_std_pre_scale , koay_result_pre, channels))
    #M_new_pre, s_new_pre = koay_test3(np.mean(img_pre_vent), back_std_pre_scale , koay_result_pre, channels)
    #print(M_new_pre, s_new_pre)
    
    #print("SNR ratio: {0:.4f}".format(koay_result_pre/koay_result_post))
    
    ##print("signal ratio: {0:.4f}".format(koay_result_pre/koay_result_post))
    
    cov_ = np.cov(np.vstack((img_post_vent, img_pre_vent)))
    #print(np.sqrt(2.0*np.trace(cov_) - np.sum(cov_)))
    mean_pre_vent = np.mean(img_pre_vent)
    mean_post_vent = np.mean(img_post_vent)
    u_means = np.sqrt((cov_[0,0]/mean_pre_vent)**2.0 + (cov_[1,1] /mean_post_vent )**2.0 - 2.0*cov_[0,1]/(mean_pre_vent * mean_post_vent))
    print(u_means)
    #koay_result_post2, err = newton_koay(img_post_vent /  back_std_post_scale , channels)
    #print(koay_result_post2, err)
    
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
    
    gs = plt.GridSpec(4,4, wspace=0.2, hspace=0.8) 
    # Create a figure
    fig = plt.figure(figsize=(9, 13))

    # SUBFIGURE 1
    # Create subfigure 1 (takes over two rows (0 to 1) and column 0 of the grid)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("{0}: VWI_pre_masked_vent bootstrap".format(case_id), fontsize = font_size)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("VWI_post_masked_vent bootstrap", fontsize = font_size)
    
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_title("post-pre bootstrap", fontsize = font_size)
    ax4 = fig.add_subplot(gs[3, :])u_e_confidence
    ax4.set_title(r'$\frac{pre_{mean}}{post_{mean}} post - pre$', fontsize = font_size)
    

    #for idx, data in enumerate(histo_grams):
        #bins = data[1]
        #hist = data[0]
        #width =  (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #mean = np.average(center, weights=hist)
        #var = np.average((center - mean)**2, weights=hist)
        #print(mean, np.sqrt(var))
        #ax.bar(center, hist, align='center', width=width, alpha = 0.5, label=labels[idx])
        #c = np.cumsum(hist)
        #cdf_list.append(c/c[-1])
    ax1.hist(pre_dist_std, bins=n_bins)
    ax1.axvline(x=np.std(img_pre_vent), color='r')
    ax2.hist(post_dist_std, bins=n_bins)
    ax2.axvline(x=np.std(img_post_vent), color='r')
    ax2.set_xlabel("stddev", fontsize = font_size)
    ax1.set_ylabel("count", fontsize = font_size)
    ax2.set_ylabel("count", fontsize = font_size)
    
    ax3.hist(post_dist_mean - pre_dist_mean, bins=n_bins)
    ax3.axvline(x=(mean_post_vent-mean_pre_vent), color='r')
    
    ax4.hist(mean_pre_vent / mean_post_vent *post_dist_mean - pre_dist_mean, bins=n_bins)
    ax4.axvline(x=0.0, color='r')
    ax3.set_xlabel("mean post - pre", fontsize = font_size)
    ax3.set_ylabel("count", fontsize = font_size)
    ax4.set_ylabel("count", fontsize = font_size)
    plt.show()
    #ax.legend()

    
    #SNR_post_vent_2  = np.mean(img_post_vent) / np.std(img_post_vent)
    #SNR_pre_vent_2  = np.mean(img_pre_vent) / np.std(img_pre_vent)
    
    #koay_result, err = newton_koay(SNR_post_vent_2, channels)

    #print("SNR : {0} ".format(SNR_post_vent_2))
    #print(koay_result, err)
    #print(koay_test(np.mean(img_pre_vent), back_std_pre_scale , SNR_post_vent))
    
    #koay_result, err = newton_koay(SNR_pre_vent_2, channels)

    #print("SNR : {0} ".format(SNR_pre_vent_2))
    #print(koay_result, err)
    
    #SNR_background_post = np.mean(img_post_back) / back_std_post_scale
    #SNR_background_pre = np.mean(img_pre_back) / back_std_pre_scale
    
    #print( "post back SNR: {0:.4f} pre back SNR {1:.4f}".format(SNR_background_post, SNR_background_pre))
    
    #test_factor = np.sqrt( (4.0 - np.pi) / np.pi)
    #SNR_back_post_est = np.mean(img_post_back) / np.std(img_post_back)
    #SNR_back_pre_est = np.mean(img_pre_back) / np.std(img_pre_back)
    
    #print( "2 post back SNR est: {0:.4f} pre back SNR est {1:.4f}".format(
            #( SNR_back_post_est), 
            #( SNR_back_pre_est ))
            #)
    #print( "2 post back SNR factor: {0:.4f} pre back SNR factor {1:.4f}".format(
            #( SNR_back_post_est * test_factor ), 
            #( SNR_back_pre_est * test_factor ))
            #)

    #skewness_factor = (4.0-np.pi)**(3.0/2.0) / (2.0*np.sqrt(np.pi)*(np.pi-3.0) )
    #print( "2 post back skewness test: {0:.4f} pre back skewness test {1:.4f}".format(
            #( stats.skew(img_post_back) * skewness_factor ), 
            #( stats.skew(img_pre_back) * skewness_factor ))
            #)
    
    #mean_post_back = np.mean(img_post_back)
    #std_post_back = np.std(img_post_back) 
    #A, sigma =  fsolve(rician_eqn, (np.mean(img_post_vent) , np.std(img_post_back)), (np.mean(img_post_vent) , np.std(img_post_back)), factor=10)
    #print(A,sigma)
    #print(rician_eqn((A, sigma), np.mean(img_post_vent) , np.std(img_post_back)))
    #r_koay = mean_post_back/std_post_back
    
    
    #koay_result = newton_koay(r_koay, channels)
    
    #print(lower_bound(channels))
    #print("SNR : {0} ".format(r_koay))
    #print(koay_result)
    
    
    
    
    #SNR_model_post = np.mean(img_post_model) / post_back_noise
    #SNR_model_pre = np.mean(img_pre_model) / pre_back_noise
    
    
    #print( "post model SNR: {0:.4f} pre model SNR {1:.4f}".format(SNR_model_post, SNR_model_pre))
    

