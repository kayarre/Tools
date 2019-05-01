import os


dir_path = '/raid/home/ksansom/caseFiles/mri/PAD_proj/case1/fluent'

file_list_name = 'file_list'

cas_file = 'TH_0_PATIENT7100_M.cas'
journal_name = 'convert_ensight.jou'

out_path = '/raid/home/ksansom/caseFiles/mri/PAD_proj/case1/fluent/ensight'


with open(os.path.join(dir_path, file_list_name), 'rU') as f:
    lines = f.readlines()

full_path_list = [os.path.join(dir_path, ln.strip()) for ln in lines]
full_out_list = [os.path.join(out_path, ln.strip()) for ln in lines]

#print(full_path_list)


read_case = '/file/read-case "{0}"\n'.format(os.path.join(dir_path, cas_file))

with open(os.path.join(dir_path, journal_name), 'w') as f:
    f.write(read_case)
    for cas, out in zip(full_path_list, full_out_list) :
        f.write('/file/read-data "{0}"\n'.format(cas))
        #f.write('/file/export/ensight-gold "{0}" absolute-pressure x-wall-shear y-wall-shear z-wall-shear () yes th_0_patient7100_m_combined_vmtk_decimate_trim2_ext2 () default_interior-1 () no "{0}"\n'.format(out))
        f.write('/file/export/ensight-gold "{0}" absolute-pressure x-wall-shear y-wall-shear z-wall-shear () yes 1 () default_interior-1 () no "{0}"\n'.format(out))
