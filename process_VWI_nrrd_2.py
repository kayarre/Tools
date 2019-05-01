import nrrd
import json
import numpy as np

json_file = "/home/sansomk/caseFiles/mri/VWI_proj/step3_normalization.json"

with open(json_file, 'r') as f:
    data = json.load(f)

labels = ["post_float","pre_float", "VWI_post_masked_vent", "VWI_post_vent",
        "pre2post", "level_set", "VWI_pre2post_masked_vent",
        "VWI_background_post_masked", "VWI_background_post",
        "VWI_background_pre_masked", "VWI_background_pre", 
        "model-label_cropped", "model_post_masked",
        "model_pre2post_masked"]

subset_labels = ["VWI_post_masked_vent", "VWI_pre2post_masked_vent",
    "VWI_background_post_masked", "VWI_background_pre_masked",
    "model_post_masked", "model_pre2post_masked"]

case_id_list = []
for case_id, images in data.items():
    case_id_list.append(case_id)

image_dict = {}

for image_label in subset_labels:
    image_dict[image_label] = {}
    print(image_label)
    for case_id in case_id_list:
        mask_path = data[case_id][image_label]   
        #vwi_mask, vwi_mask_header = nrrd.read(vwi_mask_path)
        image_tuple = nrrd.read(mask_path)
        image_dict[image_label][case_id] = image_tuple
        vwi_mask_array = image_tuple[0][image_tuple[0] >= 0.0]
        #print(image_tuple[0].size - vwi_mask_array.shape[0])
        print(case_id, "mean: {0}".format(np.mean(vwi_mask_array)),
            " std: {0}".format(np.std(vwi_mask_array)),
            " sample size {0}".format(vwi_mask_array.shape[0]))
    #print(np.std(vwi_mask_array), np.std(pre2post_mask_array))



#for case_id, images in data.items():
    #image_dict[case_id] = {}
    #print(case_id)
    #for image_label in subset_labels:
        #mask_path = images[image_label]    
        ##vwi_mask, vwi_mask_header = nrrd.read(vwi_mask_path)
        #image_tuple = nrrd.read(mask_path)
        #image_dict[case_id][image_label] = image_tuple
        #vwi_mask_array = image_tuple[0][image_tuple[0] >= 0.0]
        #print(image_label, "mean: {0}".format(np.mean(vwi_mask_array)),
            #" std: {0}".format(np.std(vwi_mask_array)),
            #" sample size {0}".format(vwi_mask_array.shape[0]))
    ##print(np.std(vwi_mask_array), np.std(pre2post_mask_array))
