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


case_id = "case8"


json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_normalization.json"


with open(json_file, 'r') as f:
    data = json.load(f)

labels = ["post_float","pre_float", "VWI_post_masked_vent", "VWI_post_vent",
        "pre2post", "level_set", "VWI_pre2post_masked_vent",
        "VWI_background_post_masked", "VWI_background_post",
        "VWI_background_pre_masked", "VWI_background_pre", 
        "model-label_cropped", "model_post_masked",
        "model_pre2post_masked",  "VWI_post_float_cropped", "VWI_background_intersection",
        "VWI_background_post_intersection", "VWI_background_pre_intersection"]

pre_path = data[case_id]["VWI_background_pre_intersection"]
post_path = data[case_id]["VWI_background_post_intersection"]


image_tuple_pre = nrrd.read(pre_path)
image_tuple_post = nrrd.read(post_path)

img_post_back_inter = image_tuple_post[0][ image_tuple_post[0]>= 0.0]
img_pre_back_inter = image_tuple_pre[0][image_tuple_pre[0] >= 0.0]

print(case_id, "post vent inter MEAN: {0:.4f} pre vent inter MEAN {1:.4f}".format(np.mean(img_post_back_inter) , np.mean(img_pre_back_inter) ))
print(case_id, "post vent inter shape: {0} pre vent inter shape {1}".format(img_post_back_inter.shape ,img_pre_back_inter.shape ))
print(case_id, "post vent inter STD: {0:.4f} pre vent inter STD {1:.4f}".format(np.std(img_post_back_inter) , np.std(img_pre_back_inter) ))    
