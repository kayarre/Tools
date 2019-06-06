import nrrd
import json
import numpy as np
import pickle
import os
from pyemd import emd
from itertools import cycle
import pandas as pd


import scipy.stats as stats
#from scipy.optimize import fsolve
#from scipy.special import iv
#from scipy.special import factorial2, factorial
#from scipy.special import hyp1f1
from sklearn import metrics
import matplotlib.pyplot as plt

import scipy.spatial.distance as distance


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
        raise  Exception("unknown enhancement measure {0}".format(kind))
        return 0.0
    
    E = post_ - pre_
    if return_parts:
        return E, post_, pre_
    else:
        return E
    
def distance_matrix(x, y, metric_name="euclidean"):
    """ calculate a distance function between two 1-D arrays
    ----------
    x : 1D numpy array
    y : 1D  numpy array
    -------
    returns the distance matrix
    """
    return distance.cdist(np.atleast_2d(x).T, np.atleast_2d(y).T, metric=metric_name)


def average_distance(x, y, metric_name="euclidean", chunksize=10**3):
    """ calculate a distance function between two 1-D arrays
    ----------
    x : 1D numpy array
    y : 1D  numpy array
    -------
    returns the distance matrix
    """
    N = x.shape[0]
    Ny = y.shape[0]
    d = 0.0
    for i in range(0, N, chunksize):
        d_mat =  distance.cdist(np.atleast_2d(x[i:i+chunksize]).T, np.atleast_2d(y).T, metric=metric_name)
        d += d_mat.sum()
    return d / (N*Ny)


def emd_function(a,b, metric="euclidean"):
    dist_mat = distance_matrix(a,b, metric=metric_name)
    emd(a.astype(float), b.astype(float), dist_mat) 
    return emd(a.astype(float), b.astype(float), dist_mat)

def hellinger(a,b):
    """ calculate hellinger between two distributions
    ----------
    a : histogram counts or pdf of size N, numpy array
    b : histogram counts or pdf of size N, numpy array
     assume normalized input
    -------
    returns the hellinger distance,
    """
    return 1.0/np.sqrt(2.0)*np.sqrt( np.sum( (np.sqrt(a) - np.sqrt(b) )**2.0 ) )

def kullbackleiber(a,b):
    #same as scipy.stats.entropy(p, q)
    """ calculate Kullback Leibler divergence. between two distributions
    ----------
    a : histogram counts or pdf of size N, numpy array
    b : histogram counts or pdf of size N, numpy array
     assume normalized input
    -------
    returns the Kullback Leibler divergence.
    """

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def bhattacharyya_coefficient(a, b):
    """ Determine the bhattacharyya coefficient.
    ----------
    a : histogram counts or pdf of size N, numpy array
    b : histogram counts or pdf of size N, numpy array
    -------
    returns the bhattacharyya coefficient
    """
    return np.sum(np.sqrt(a*b))


def bhattacharyya(a, b):
    """Determine the bhattacharyya distance."""
    return -np.log(bhattacharyya_coefficient(a, b))

def intersection(a,b):
    minima = np.minimum(a, b)
    intersection = np.true_divide(np.sum(minima), np.sum(a))
    return intersection

def non_intersection(a,b):
    minima = np.minimum(a, b)
    # n - sum(min(a_i, b_i))
    # n is the total number of things that are binned
    return a.sum() - np.sum(minima)

def matusita(a,b):
   return np.sqrt(np.sum( np.power(np.sqrt(a) - np.sqrt(b), 2)))

def intensitydistance(dist_matrix, metric):
    
    if (metric == "nearestneighbor"):
        return dist_matrix.min()
    elif (metric == "furthestneighbor"):
        return dist_matrix.max()
    else:
        raise  Exception("unknown distance measure {0}".format(metric))
        return -999999.0

def minimum_difference_pair_assignments(a,b, distance_metric="cityblock"):
    """
    https://cedar.buffalo.edu/~srihari/papers/PRJ02-DM.pdf
    On measuring the distance between histograms the ordinal case
    """

    #prefixsum = np.zeros_like(a)
    #h_dist = 0.0
    #for i in range(a.shape[0]):
        #prefixsum[i] = distance.cityblock(a[0:i+1], b[0:i+1]).sum()
        #h_dist += np.abs(prefixsum)
    #return h_dist

    prefixsum = 0.0
    h_dist = 0.0
    for i in range(a.shape[0]):
        prefixsum += a[i] - b[i]
        h_dist += np.abs(prefixsum)
    return h_dist


def distance_function(data,  metric="default", normalize=None):
    """ calculate distance between two distributions
    ----------
    data: tuple of size 4, (post_hist, pre_hist, post_pre)
    metric: the metric of the distance metric
    normalize: normalize the inputs
    -------
    returns the distance calculation
    """
    
    if (data[0][0].shape != data[1][0].shape):
        raise  Exception(" Histograms a and b not the same size a:{0} b:{1}".format(data[0][0].shape[0], data[1][0].shape[0]))
        return 99999.0
    elif (data[2].shape != data[3].shape):
        raise  Exception(" input data a and b not the same size a:{0} b:{1}".format(data[2].shape[0], data[3].shape[0]))
        return 99999.0
    
    #normalize_measures = ["euclidean", "correlation", "manhattan","braycurtis", "hellinger", "kullbackleiber"]
    
    #if ( normalize == True or (normalize == None and  name in normalize_measures)):
        #max_a = np.max(hista.shape[0])
        #max_b = np.max(histb.shape[0])
        #a = a / max_a
        #b = b / max_b
    hista = data[0][0]
    histb = data[1][0]
    binsa = data[0][1]
    binsb = data[1][1]

    binctr_a = 0.5*(binsa[1:]+binsa[:-1])
    binctr_b = 0.5*(binsb[1:]+binsb[:-1])

    if (metric == "default" or metric == "euclidean"):
        dist = distance.euclidean(hista, histb)
    elif(metric == "manhattan"):
        dist = distance.cityblock(hista, histb)
    elif(metric == "nonintersection"):
        dist = non_intersection(hista, histb)
    elif(metric == "kullbackleiber"):
        "Kullback-Leibler divergence D(P || Q) for discrete distributions"
        dist = stats.entropy(hista, histb)# switched for test
    elif(metric == "bhattacharyya"):
        dist = bhattacharyya(hista, histb)
    elif(metric == "matusita"):
        dist = matusita(hista, histb )
    #elif(metric in ["nearestneighbor", "furthestneighbor"]):
        #cdist_calc = distance_matrix(data[2], data[3], metric_name="cityblock")
        #dist = intensitydistance(cdist_calc, metric)
    elif(metric == "meandistance"):
        dist = np.abs(data[2].mean() - data[3].mean())
    elif(metric == "averagedistance"):
        dist = average_distance(data[2], data[3], metric_name="cityblock", chunksize=10**2 )
    elif(metric == "jensenshannon"):
        dist = distance.jensenshannon(hista, histb)
    elif(metric == "correlation"):
        dist = distance.correlation(hista, histb)
    elif(metric =="braycurtis"):
        dist =distance.braycurtis(hista, histb)
    elif(metric == "hellinger"):
        dist = hellinger(hista, histb)
    elif(metric == "wasserstein"):
        dist = stats.wasserstein_distance(binctr_a, binctr_b,  u_weights=hista, v_weights=histb)
    elif(metric == "energy"):
        dist = stats.energy_distance(binctr_a, binctr_b, u_weights=hista, v_weights=histb)
    elif(metric == "kolmogorovsmirnov"):
        res = stats.ks_2samp(hista, histb)
        dist = res.statistic
    elif(metric == "additivechi2"):
        res = metrics.pairwise.additive_chi2_kernel(np.atleast_2d(hista), np.atleast_2d(histb))
        dist = res[0][0]
    elif(metric == "MPDA"):
        dist = minimum_difference_pair_assignments(hista, histb)
    else:
        raise  Exception("unknown distance measure {0}".format(metric))

    #elif(metric == "earthmovers"):
    #    dist = emd_function(a,b)
    return dist

def get_histograms(post, pre, density_bool=False):
    range_b = [np.min([pre.min(), post.min()]), np.max([pre.max(), post.max()])]
    nbins = np.histogram_bin_edges(post, range=range_b, bins='sqrt')
    post_hist = np.histogram(post, range=range_b, bins=nbins, density=density_bool)
    pre_hist = np.histogram(pre, range=range_b, bins=nbins, density=density_bool)
    return post_hist, pre_hist

#vars

#conf = 2.0 #95% confidence interval
#percent_boot = 0.01
font_size = 20

#figure_1 = True
#figure_2 = True

# all the distance comparisons require the distributions to have the same support
# same bins and same bin widths
# removed  three

distance_features = ["manhattan", "euclidean", "nonintersection", "bhattacharyya",
                                  "matusita",   "meandistance", "averagedistance",
                                  "jensenshannon", "correlation", "braycurtis",  "hellinger", "wasserstein",
                                  "energy", "kolmogorovsmirnov", "additivechi2", "MPDA"
                                  ] #  "nearestneighbor", "furthestneighbor", "kullbackleiber",

#distance_features = ["euclidean", "jensenshannon", "correlation", "bhattacharyya"
                                  #"manhattan", "braycurtis",  "hellinger", "wasserstein"]

json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_normalization.json"

#pickle_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_pickle.pkl"
write_file_dir = "/home/sansomk/caseFiles/mri/VWI_proj"
write_dir = "VWI_analysis"
plots_dir = "plots"
#overwrite = 0
#overwrite_out = False
#skip_bootstrap = True

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

#image_dict = {}
#channels = int(1)
#image_path_list = []
#bootstrap_fig_list = []
case_id_list = []
enhancement_type =[]
feature_type = []
dist_type = []
distance_list = []
dist_norm_list = []

imtype = cycle(["pre", "post"])
etype = cycle(["E", "E1", "E2", "E3", "E4"])


for case_id, paths in data.items():
    try:
        # Create target Directory
        test = os.path.join(write_file_dir, case_id, plots_dir)
        os.mkdir(test)
        print("Directory " , test ,  " Created ") 
    except FileExistsError:
        print("Directory " , test ,  " already exists")
        
    
    img_post_vent = nrrd.read(paths["VWI_post_masked_vent"])[0]
    img_post_vent = img_post_vent[img_post_vent >= 0.0]
    eta_model = nrrd.read(paths["model_post_masked"])[0] 
    eta_model = eta_model[eta_model >= 0.0]   
    
 
    img_pre_vent = nrrd.read(paths["VWI_pre2post_masked_vent"])[0]
    img_pre_vent = img_pre_vent[img_pre_vent >= 0.0]
    xi_model = nrrd.read(paths["model_pre2post_masked"])[0] 
    xi_model = xi_model[xi_model >= 0.0]
    

    eta_PI = case_imgs["VWI_post_PI_masked"][0][case_imgs["VWI_post_PI_masked"][0] >= 0.0]
    xi_PI = case_imgs["VWI_pre2post_PI_masked"][0][case_imgs["VWI_pre2post_PI_masked"][0] >= 0.0]
    
    #back_std_xi = np.std(img_pre_back) 
    #back_std_eta = np.std(img_post_back)
    
    ## ventricle portion
    cov_vent = np.cov(np.vstack((img_post_vent, img_pre_vent)))
    N_vent = float(img_post_vent.shape[0])
    #cov_back = np.cov(np.vstack((img_post_back_inter, img_pre_back_inter)))
    #print(np.sqrt(2.0*np.trace(cov_vent) - np.sum(cov_vent)))

    eta_vent = np.mean(img_post_vent) # mean ventricle post
    xi_vent = np.mean(img_pre_vent) # mean ventricle pre
    u_eta_vent_2 = cov_vent[0,0] /  N_vent  # uncertainty vent post square
    u_xi_vent_2 = cov_vent[1,1] /  N_vent  # uncertainty vent pre square
    u_eta_xi_vent = cov_vent[0,1] / N_vent # covariance between pre and post  
    
    E_vent, E_post_vent, E_pre_vent = VWI_Enhancement(img_post_vent, img_pre_vent, 1.0, 1.0, kind = "E1", return_parts=True)
    E_post_vent_hist, E_pre_vent_hist = get_histograms(E_post_vent, E_pre_vent, density_bool=True)

    E1_vent, E1_post_vent, E1_pre_vent = VWI_Enhancement(img_post_vent, img_pre_vent, eta_vent, xi_vent, kind = "E1",
                                                         return_parts=True)
    E1_post_vent_hist, E1_pre_vent_hist = get_histograms(E1_post_vent, E1_pre_vent, density_bool=True)    

    E2_vent, E2_post_vent, E2_pre_vent = VWI_Enhancement(img_post_vent, img_pre_vent, eta_vent, xi_vent, kind = "E2",
                                                         return_parts=True)
    E2_post_vent_hist, E2_pre_vent_hist = get_histograms(E2_post_vent, E2_pre_vent, density_bool=True)

    E3_vent, E3_post_vent, E3_pre_vent= VWI_Enhancement(img_post_vent, img_pre_vent, eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    E3_post_vent_hist, E3_pre_vent_hist = get_histograms(E3_post_vent, E3_pre_vent, density_bool=True)


    E4_vent, E4_post_vent, E4_pre_vent= VWI_Enhancement(img_post_vent, img_pre_vent, eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    E4_post_vent_hist, E4_pre_vent_hist = get_histograms(E4_post_vent, E4_pre_vent, density_bool=True)
    
    

    
    E_model, E_post_model, E_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                           1.0, 1.0, kind = "E1", return_parts=True)
    E_post_model_hist, E_pre_model_hist = get_histograms(E_post_model, E_pre_model, density_bool=True)

    E1_model, E1_post_model, E1_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E1", return_parts=True)
    E1_post_model_hist, E1_pre_model_hist = get_histograms(E1_post_model, E1_pre_model, density_bool=True)

    E2_model, E2_post_model, E2_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E2", return_parts=True)
    E2_post_model_hist, E2_pre_model_hist = get_histograms(E2_post_model, E2_pre_model, density_bool=True)

    E3_model, E3_post_model, E3_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    E3_post_model_hist, E3_pre_model_hist = get_histograms(E3_post_model, E3_pre_model, density_bool=True)

    E4_model, E4_post_model, E4_pre_model = VWI_Enhancement(eta_model, xi_model,
                                                            eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    E4_post_model_hist, E4_pre_model_hist= get_histograms(E4_post_model, E4_pre_model, density_bool=True)
    
    #E_PI, E_post_PI, E_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
    #                                                       1.0, 1.0, kind = "E1", return_parts=True)
    
    E_PI, E_post_PI, E_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           1.0, 1.0, kind = "E1", return_parts=True)
    E_post_PI_hist, E_pre_PI_hist = get_histograms(E_post_PI, E_pre_PI, density_bool=True)


    E1_PI, E1_post_PI, E1_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           eta_vent, xi_vent, kind = "E1", return_parts=True)
    E1_post_PI_hist, E1_pre_PI_hist = get_histograms(E1_post_PI, E1_pre_PI, density_bool=True)

    E2_PI, E2_post_PI, E2_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                           eta_vent, xi_vent, kind = "E2", return_parts=True)
    E2_post_PI_hist, E2_pre_PI_hist = get_histograms(E2_post_PI, E2_pre_PI, density_bool=True)

    E3_PI, E3_post_PI, E3_pre_PI = VWI_Enhancement(eta_PI, xi_PI,
                                                            eta_vent, xi_vent, kind = "E3",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    E3_post_PI_hist, E3_pre_PI_hist = get_histograms(E3_post_PI, E3_pre_PI, density_bool=True)

    E4_PI, E4_post_PI, E4_pre_PI  = VWI_Enhancement(eta_PI, xi_PI,
                                                            eta_vent, xi_vent, kind = "E4",
                                                        std_post_vent = np.sqrt(u_eta_vent_2), 
                                                        std_pre_vent = np.sqrt(u_xi_vent_2),
                                                        return_parts=True)
    E4_post_PI_hist, E4_pre_PI_hist= get_histograms(E4_post_PI, E4_pre_PI, density_bool=True)


    list_vent = {"ventricle": [(E_post_vent_hist, E_pre_vent_hist, E_post_vent, E_pre_vent),
                                            (E1_post_vent_hist, E1_pre_vent_hist, E1_post_vent, E1_pre_vent ),
                                            (E2_post_vent_hist, E2_pre_vent_hist, E2_post_vent, E2_pre_vent ),
                                            (E3_post_vent_hist, E3_pre_vent_hist, E3_post_vent, E3_pre_vent ),
                                            (E4_post_vent_hist, E4_pre_vent_hist, E4_post_vent, E4_pre_vent)],
                       "wall":  [(E_post_model_hist, E_pre_model_hist, E_post_model, E_pre_model),
                                    (E1_post_model_hist, E1_pre_model_hist, E1_post_model, E1_pre_model ),
                                    (E2_post_model_hist, E2_pre_model_hist, E2_post_model, E2_pre_model),
                                    (E3_post_model_hist, E3_pre_model_hist, E3_post_model, E3_pre_model),
                                    (E4_post_model_hist, E4_pre_model_hist, E4_post_model, E4_pre_model)]
                        "Pituitary_Infundibulum":  [(E_post_PI_hist, E_pre_PI_hist, E_post_PI, E_pre_PI),
                                    (E1_post_PI_hist, E1_pre_PI_hist, E1_post_PI, E1_pre_PI ),
                                    (E2_post_PI_hist, E2_pre_PI_hist, E2_post_PI, E2_pre_PI),
                                    (E3_post_PI_hist, E3_pre_PI_hist, E3_post_PI, E3_pre_PI),
                                    (E4_post_PI_hist, E4_pre_PI_hist, E4_post_PI, E4_pre_PI)]
                       }
                        

    #dist = 0.0
    for feature, feature_list in list_vent.items():
        count = int(0)
        distance_norm = {}
        for data_ in feature_list:
            #iim = next(imtype)
            e = next(etype)
            for dist_f in distance_features:
                dist = distance_function(data_, metric = dist_f, normalize=False)
                if ( count == 0 ):
                    distance_norm[dist_f] = dist
                    #print(dist, distance_norm[dist_f])
                    dist_norm_list.append(dist / distance_norm[dist_f])
                else:
                    dist_norm_list.append(dist / distance_norm[dist_f])

                case_id_list.append(case_id)
                enhancement_type.append(e)
                dist_type.append(dist_f)
                distance_list.append(dist)
                
                feature_type.append(feature)
            
            count += 1


distance_from_E = [  x - 1.0 for x in dist_norm_list]

df = pd.DataFrame({"case_id": case_id_list, "enhancement_type": enhancement_type, "distance_metric":dist_type, "feature":feature_type, "distance":distance_list, "distance_norm": dist_norm_list, "distance_from_E": distance_from_E})

df.to_pickle(os.path.join(write_file_dir, "distance_metrics_df.pkl"))



