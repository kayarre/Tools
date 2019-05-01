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




write_file_dir = "/home/sansomk/caseFiles/mri/VWI_proj"
write_dir = "VWI_analysis"

cases = ["case1", "case3", "case4", "case5", "case7", "case8", "case12", "case13", "case14"]


file_names = ["E.nrrd", "E1.nrrd", "E3.nrrd", "E5.nrrd"]

json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_normalization.json"


write_dir = "VWI_analysis"
plots_dir = "plots"

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



font_size = 20

#test = np.linspace(1,64,64, dtype=np.int)
#lb_test = lower_bound(test)
#plt.plot(test, lb_test)
#plt.show()

channels = int(1)
image_path_list = []
bootstrap_fig_list = []
for case_id  in cases:
    post_model_path = data[case_id]["model_post_masked"] #[0][case_imgs["model_post_masked"][0] >= 0.0]
    #pre_model_path = data[case_id]["model_pre2post_masked"] #0][case_imgs["model_pre2post_masked"][0] >= 0.0]
    
    post_model = nrrd.read(post_model_path)
    #pre_model = nrrd.read(pre_model_path)
    pi_post_file = data[case_id]["VWI_post_PI_masked"]
    pi_pre_file = data[case_id]["VWI_pre2post_PI_masked"]
    
    pi_post = nrrd.read(pi_post_file)
    pi_pre = nrrd.read(pi_pre_file)
     
    mean_post_pi = pi_post[0].mean()
    mean_pre_pi = pi_pre[0].mean()
    
    E_pi = mean_post_pi - mean_pre_pi
                            

    try:
        # Create target Directory
        test = os.path.join(write_file_dir, case_id, plots_dir)
        os.mkdir(test)
        print("Directory " , test ,  " Created ") 
    except FileExistsError:
        print("Directory " , test ,  " already exists")
    
    gs3 = plt.GridSpec(1,1, wspace=0.2, hspace=0.8)
    fig3 = plt.figure(figsize=(13, 8))
    ax1_3 = fig3.add_subplot(gs3[0, 0])
    ax1_3.set_title("Compare Histograms", fontsize = font_size)
    
    ax1_3.set_xlabel(r'Normalized Intensity: {0}'.format(case_id), fontsize = font_size)
    ax1_3.set_ylabel(r'Frequency', fontsize = font_size)
    



    E_files = {}
    for e in file_names:
        path_to_e = os.path.join(write_file_dir, case_id, write_dir, e)
        file_test = nrrd.read(path_to_e)
        E_files[e] = file_test[0][post_model[0] >=0]

        ax1_3.hist(E_files[e], histtype='step', label=e.split('.')[0], bins=100, density=True)


    path_E3 = os.path.join(write_file_dir, case_id, plots_dir, "Compare_Hist_E.png")
    fig3.legend(fontsize=font_size)
    #fig3.show()
    
    fig3.savefig(path_E3)
    #input("Press Enter to continue...")
    #fig3.close()
    #del fig3


print("hooray")
print(image_path_list)
print(bootstrap_fig_list)

    

