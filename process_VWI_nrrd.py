import nrrd
import json
import numpy as np

json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step2_normalization.json"

with open(json_file, 'r') as f:
    data = json.load(f)
    

for case_id, images in data.items():
    vwi_path = images['post_float']
    vwi_mask_path = images['VWI_post_float_ventricle_masked']
    pre2post_path = images['pre2post']
    pre2post_mask_path = images['pre2post_ventricle_masked']
    
    
    vwi_mask, vwi_mask_header = nrrd.read(vwi_mask_path)
    pre2post_mask, pre2post_mask_header = nrrd.read(pre2post_mask_path)
    #print(case_id)
    #print(vwi_mask.shape, pre2post_mask.shape)
    
    vwi_mask_array = vwi_mask[vwi_mask >= 0.0]
    pre2post_mask_array = pre2post_mask[pre2post_mask >= 0.0]
    print(case_id, "post mean: {0}".format(np.mean(vwi_mask_array)), "pre mean: {0}".format(np.mean(pre2post_mask_array)), "post std: {0}".format(np.std(vwi_mask_array)), "pre std: {0}".format(np.std(pre2post_mask_array)))
    #print(np.std(vwi_mask_array), np.std(pre2post_mask_array))
