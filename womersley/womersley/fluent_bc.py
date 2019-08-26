from mako.template import Template
from shutil import copyfile

bcs = {}
for index, row in br_file.iterrows():
    bcs[row['Label']] = {"X" : row['X']/1000.0, "Y" : row['Y']/1000.0, "Z" : row['Z']/1000.0,
                         "BoundaryRadius" : row['BoundaryRadius']/1000.0,
                        "R": row["R"], "C": row["C"], "P":11000, 'direction': row["direction"],
                        "coef_file": row["fourier_fn"]}
    
    
vel_params = False 
for bc_name in bcs.keys():
    if( bcs[bc_name]['direction'] != int(0)):
        if not vel_params: # only do params file once
            wom_header = Template(filename=os.path.join(mako_path, 'parameters.h.mako'))
            #print(wom_header.render(waveform=waveform_2))
            wom_h_text = wom_header.render(waveform=waveform)
            test_path = os.path.join(tmp_path, "parameters.h")
            with open(test_path, 'w') as wom_h_file:
                wom_h_file.write(wom_h_text)
            vel_params = True
        
        wom_c_tmp = Template(filename=os.path.join(mako_path, 'inlet_w_vel.c.mako'))
        #print(RC_c_tmp.render(bc_name=bc_name))
        wom_text = wom_c_tmp.render(bc=bcs[bc_name])
        
        test_path = os.path.join(tmp_path, "inlet_w_vel_{0}.c".format(bc_name))
        with open(test_path, 'w') as wom_file:
            wom_file.write(wom_text)
    else:
        RC_c_tmp = Template(filename=os.path.join(mako_path, 'RC.c.mako'))
        #print(RC_c_tmp.render(bc_name=bc_name))
        RC_c_text = RC_c_tmp.render(bc_name=bc_name)

        test_path = os.path.join(tmp_path, "RC_{0}.c".format(bc_name))
        with open(test_path, 'w') as RC_c:
            RC_c.write(RC_c_text)

#constant files that are only needed once
file_list = [ "complex_ops.h.mako", "complex_rec.h.mako", "zbes.h.mako"]

for file_n in file_list:
    const_tmp = Template(filename=os.path.join(mako_path, file_n ))
    const_text = const_tmp.render()

    test_path = os.path.join(tmp_path, os.path.splitext(file_n)[0])
    with open(test_path, 'w') as const_file:
        const_file.write(const_text)


for wv in [waveform]:
    copyfile(os.path.join(wv.dir_path, waveform.waveform_ctr_name), 
             os.path.join(tmp_path, waveform.waveform_ctr_name))
    
    

wall = [4]
velocity = [9]
for i in range(3,len(bcs.keys())+4):
    if( i in wall):
        print("/define/boundary-conditions/modify-zones/zone-type {0} wall ()".format(i))
    elif(i in velocity):
        print("/define/boundary-conditions/modify-zones/zone-type {0} velocity-inlet ()".format(i))
    else:
        print("/define/boundary-conditions/modify-zones/zone-type {0} pressure-outlet ()".format(i))
        
        
count = 0
for bc_name in bcs.keys():
    if( bcs[bc_name]['direction'] != int(0) ):
        continue
    else:
        if (count == 0):
            print('/define/user-defined/compiled-functions compile libRC_{0} yes "RC_{0}.c" "" "RC.h" "" ()'.format(bc_name))
            print('/define/user-defined/compiled-functions load libRC_{0}'.format(bc_name))
        else: 
            print('/define/user-defined/compiled-functions compile libRC_{0} yes y "RC_{0}.c" "" y "RC.h" "" ()'.format(bc_name))
            print('/define/user-defined/compiled-functions load libRC_{0}'.format(bc_name))
        count += 1
        
        
count = 0
for bc_name in bcs.keys():
    if(  bcs[bc_name]['direction'] != int(0)):
        print('/define/boundary-conditions/set/velocity-inlet {1}:{0} () velocity-spec no no yes vmag yes yes udf "prox_art_vel::libWomersley_{0}" ()'.format(bc_name,tecplot_case))
    else:
        print('/define/boundary-conditions/set/pressure-outlet {1}:{0} () p-backflow-spec-gen yes  direction-spec no yes gauge-pressure yes yes udf "vein_p::libRC_{0}" ()'.format(bc_name,tecplot_case))
