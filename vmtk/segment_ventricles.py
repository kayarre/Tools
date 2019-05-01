
import os
from vmtk import vmtkscripts
import vtk
import pickle
import json

main_dir = "/home/sansomk/caseFiles/mri/VWI_proj"
store_name = "step1_normalization"
normalization_dir ="normalization"
case_list =  [1,3,4,5,7,8,11,12,13,14]
completed_cases = [1,3]

image_vols_dir = "image_vols"
registration_dir = "registration"

post_name = "VWI_post.nrrd"
post_name_float = "VWI_post_float.nrrd"
pre_name = "pre2post.nrrd"

level_set_name = "VWI_post_vent_levelset.nrrd"

case_name = "case"

VWI_float_key = "post_float"
level_set_key = "level_set"
pre2post_key = "pre2post"

id_list = [id for id in case_list if id not in completed_cases]

case_dict = {}
do_processing = False

for id in id_list:
    case_id = "{0}{1}".format(case_name, id)
    print( "segment " + case_id)
    case_dict[case_id]={}
    
    directory_path = os.path.join(main_dir, case_id)
    image_vols_path = os.path.join(directory_path, image_vols_dir)
    registration_path = os.path.join(directory_path, registration_dir)
    norm_path = os.path.join(directory_path, normalization_dir)
    
    float_path = os.path.join(image_vols_path, post_name_float)
    level_set_path = os.path.join(norm_path, level_set_name)
    
    initial_path = os.path.join(image_vols_path, post_name)
    #print(directory_path, image_vols_path, registration_path, norm_path)
    #print(initial_path)
    
    if do_processing:
        if not os.path.exists(norm_path):
            try:
                os.mkdir(norm_path)
            except:
                print("unable to create directory{0}".fomat(norm_path))
                pass

        image_reader = vmtkscripts.vmtkImageReader()
        
        image_reader.InputFileName = initial_path
        image_reader.Execute()
        
        cast_float = vtk.vtkImageCast()
        cast_float.SetInputData(image_reader.Output)
        cast_float.SetOutputScalarTypeToFloat()
        cast_float.Update()
        float_image = cast_float.GetOutput()
        #cast_float = vmtkscripts.vmtkImageCast()
        #cast_float.Image = image_reader.Output
        #cast_float.OutputType = 'float'
        #cast_float.Execute()
        
        #print(cast_float.Image.GetOrigin(), cast_float.Image.GetSpacing())
        
        # write VWI to float
        write_float = vmtkscripts.vmtkImageWriter()
        write_float.Image = float_image
        write_float.RasToIjkMatrixCoefficients = image_reader.RasToIjkMatrixCoefficients

        write_float.ApplyTransform = 1
        write_float.OutputFileName = float_path
        write_float.Execute()
        
        vmtk_seg = vmtkscripts.vmtkLevelSetSegmentation()
        vmtk_seg.Image = float_image
        vmtk_seg.Execute()
        
        # write level set 
        image_writer = vmtkscripts.vmtkImageWriter()
        image_writer.Image = vmtk_seg.LevelSetsOutput
        image_writer.RasToIjkMatrixCoefficients = image_reader.RasToIjkMatrixCoefficients
        
        image_writer.ApplyTransform = 1
        image_writer.OutputFileName = level_set_path
        image_writer.Execute()
        
        #del vmtk_seg
    else:
        print("skip processing")
    
    
    #save float VWI path
    case_dict[case_id][VWI_float_key] = float_path
    #save levelset path
    case_dict[case_id][level_set_key] = level_set_path
    
    #pre2post path
    pre2post_path = os.path.join(directory_path, registration_dir, pre_name)
    case_dict[case_id][pre2post_key] = pre2post_path
    if not os.path.exists(pre2post_path):
        print("this file doesn't exist which is a bummer: {0}".format(pre2post_path))
    print( "end segment " + case_id)
print(case_dict)

#save the different file paths
pickle.dump(case_dict,  open(os.path.join(main_dir, "{0}.pkl".format(store_name)), "wb"))
json.dump(case_dict,  open(os.path.join(main_dir, "{0}.json".format(store_name)), "w", encoding="utf8"),
                           sort_keys=False, indent=4)
#vmtklevelsetsegmentation -ifile ./case3/image_vols/VWI_post.nrrd -ofile ./case3/normalization/VWI_post_vent_levelset.nrrd
