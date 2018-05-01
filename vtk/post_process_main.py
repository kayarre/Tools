import os
import vtk
import numpy as np
from ensight2vtk_single_encas import ensight2vtk
from post_proc_cfd import post_proc_cfd

def run_script():
    #case_list = ["case1", "case3", "case4", "case5", "case7"]
    #case_list = ["case1", "case8", "case12"]
    #case_list = ["case13"]
    case_list = ["case1", "case3", "case4", "case5", "case7", "case8", "case12", "case13", "case14"]
    dir_path = "/raid/sansomk/caseFiles/mri/VWI_proj/"

    ensight_dir = "ensight"
    fluent_dir = "fluent_dsa"
    #fluent_dir = "fluent_dsa_2" # tested cell based calculation on case 4
    vtk_out = "vtk_out"
    vtk_file_1 = "wall_outfile_node.vtu"
    vtk_file_2 = "inlet_outfile_node.vtu"
    vtk_file_3 = "interior_outfile_node.vtu"

    wall = True
    surface = True
    interior = True
    file_pattern = "_dsa_0.15_fluent.encas"

    for case in case_list:
        print(case)
        head_dir = os.path.join(dir_path, case, fluent_dir)

        ensight_path = os.path.join(head_dir, ensight_dir)
        ensight_file = "{0}{1}".format(case, file_pattern)
        out_dir = os.path.join(head_dir, vtk_out)

        ensight2vtk(ensight_path, out_dir, ensight_file,
                    vtk_file_1, vtk_file_2, vtk_file_3,
                    wall, surface, interior)

        post_proc_cfd(out_dir, vtk_file_1,"point",
                      "calc_test_node.vtu", "calc_test_node_stats.vtu", N_peak=9)

def rerun_cfd_analysis():
    #case_list = ["case1"]
    case_list = case_list = ["case1", "case3", "case4", "case5", "case7", "case8", "case12", "case13", "case14"]
    dir_path = "/raid/sansomk/caseFiles/mri/VWI_proj/"

    fluent_dir = "fluent_dsa"
    vtk_out = "vtk_out"
    vtk_file_1 = "wall_outfile_node.vtu"
    #vtk_file_2 = "inlet_outfile_node.vtu"

    wall = True
    surface = True
    file_pattern = "_dsa_0.15_fluent.encas"

    for case in case_list:
        print(case)
        head_dir = os.path.join(dir_path, case, fluent_dir)
        out_dir = os.path.join(head_dir, vtk_out)
        post_proc_cfd(out_dir, vtk_file_1,"point",
                      "calc_test_node.vtu", "calc_test_node_stats.vtu", N_peak=9)

def run_convert_ensight():
    #case_list = ["case1", "case3", "case4", "case5", "case7"]
    case_list = ["case1"]
    dir_path = "/raid/sansomk/caseFiles/mri/VWI_proj/"

    ensight_dir = "ensight"
    fluent_dir = "fluent_dsa"
    #fluent_dir = "fluent_dsa_2" # tested cell based calculation on case 4
    vtk_out = "vtk_out"
    vtk_file_1 = "wall_outfile_node.vtu"
    vtk_file_2 = "inlet_outfile_node.vtu"
    vtk_file_3 = "interior_outfile_node.vtu"

    wall = True
    inlet = True
    interior = False
    file_pattern = "_dsa_0.15_fluent.encas"

    for case in case_list:
        print(case)
        head_dir = os.path.join(dir_path, case, fluent_dir)

        ensight_path = os.path.join(head_dir, ensight_dir)
        ensight_file = "{0}{1}".format(case, file_pattern)
        out_dir = os.path.join(head_dir, vtk_out)

        ensight2vtk(ensight_path, out_dir, ensight_file,
                    vtk_file_1, vtk_file_2, vtk_file_3,
                    wall, inlet, interior)




if ( __name__ == '__main__' ):
    #run_script()
    #run_convert_ensight()
    rerun_cfd_analysis()
