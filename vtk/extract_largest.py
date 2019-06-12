import vtk
#import numpy as np
import os

vmtk_avail = True
try:
    from vmtk import vtkvmtk
except ImportError:
    print("unable to import vmtk module")
    vmtk_avail = False

file_path1 = "/home/ksansom/caseFiles/mri/PAD_proj/case1/vmtk/TH_0_PATIENT7100_M.ply"
out_path = "/home/ksansom/caseFiles/mri/PAD_proj/case1/vmtk/TH_0_PATIENT7100_M_clean.vtp"

path_str = os.path.split(out_path)
split_ext = os.path.splitext(path_str[-1])

print(split_ext)
print('Reading PLY surface file.')
reader1 = vtk.vtkPLYReader()
reader1.SetFileName(file_path1)
reader1.Update()

print("extract surface mesh into connected components")
extract = vtk.vtkPolyDataConnectivityFilter()
extract.SetInputConnection(reader1.GetOutputPort())
extract.SetExtractionModeToAllRegions()
extract.GetExtractionMode()
extract.ColorRegionsOn()
extract.Update()

n_regions = extract.GetNumberOfExtractedRegions()
extract.GetOutput().GetPointData().SetActiveScalars("RegionId")

writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputConnection(extract.GetOutputPort())
writer.SetFileName(out_path)
writer.Update()

region_list = []
#surface_exlusion_list = [11, 17]
#surface_0 = vtk.vtkPolyData()
mesh_files = {}
for i in range(n_regions):
    region_list.append(i)
    thresh = vtk.vtkThreshold()
    thresh.SetInputConnection(extract.GetOutputPort())
    thresh.SetInputArrayToProcess(0, 0, 0,
                                  vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                  "RegionId")
    thresh.ThresholdBetween(i, i)
    thresh.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputConnection(thresh.GetOutputPort())
    surfer.Update()
    #vtkButterflySubdivisionFilter subdivide
    subdivide = vtk.vtkLoopSubdivisionFilter()
    subdivide.SetInputConnection(surfer.GetOutputPort())
    subdivide.SetNumberOfSubdivisions(2)
    #subdivide.Update()

    #fill_holes = vtk.vtkFillHolesFilter()
    #fill_holes.SetInputConnection(subdivide.GetOutputPort())
    #fill_holes.SetHoleSize(1000.0);
    #fill_holes.Update()
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(subdivide.GetOutputPort())
    triangle.PassLinesOff()
    triangle.PassVertsOff()
    triangle.Update()
    if(vmtk_avail):
        #creates round outlet region caps
        capper = vtkvmtk.vtkvmtkSmoothCapPolyData()
        capper.SetInputConnection(triangle.GetOutputPort())
        capper.SetConstraintFactor(1.0)
        capper.SetNumberOfRings(8)
    else:
        capper = vtk.vtkFillHolesFilter()
        capper.SetInputConnection(triangle.GetOutputPort())
        capper.SetHoleSize(1000.0);

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(capper.GetOutputPort())
    cleaner.Update()

    n_cells = cleaner.GetOutput().GetNumberOfCells()
    print("surface cells", n_cells)

    # Setup the colors array
    colors = vtk.vtkIntArray()
    colors.SetNumberOfComponents(1)
    colors.SetNumberOfValues(n_cells)
    colors.SetName("RegionId")
    for k in range(n_cells):
        colors.SetValue(k, i)

    cleaner.GetOutput().GetCellData().SetScalars(colors)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaner.GetOutputPort())
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()


    #writer2 = vtk.vtkXMLPolyDataWriter()
    #writer2.SetInputConnection(normals.GetOutputPort())
    #file_name = os.path.join(path_str[0], "{0}_{1}.vtp".format(split_ext[0], i))
    #writer2.SetFileName(file_name)
    #writer2.Write()

    #writer4 = vtk.vtkSTLWriter()
    #writer4.SetInputConnection(normals.GetOutputPort())
    #file_name = os.path.join(path_str[0], "{0}_{1}.stl".format(split_ext[0], i))
    #writer4.SetFileName(file_name)
    #writer4.Write()

    writer3 = vtk.vtkPLYWriter()
    writer3.SetInputConnection(normals.GetOutputPort())
    file_name = os.path.join(path_str[0], "{0}_{1}.ply".format(split_ext[0], i))
    writer3.SetFileName(file_name)
    writer3.Write()

    mesh_files[i]  = normals.GetOutput()
import sys
sys.exit()
#booleanOperationFilter = vtk.vtkBooleanOperationPolyDataFilter()
#        booleanOperationFilter.SetInputData(0,surface_0)
#        booleanOperationFilter.SetInputData(1,normals.GetOutput())
#        booleanOperationFilter.SetOperationToUnion()
#        booleanOperationFilter.SetTolerance(1.0e-7)
#        booleanOperationFilter.Update()

#        surface_0 = booleanOperationFilter.GetOutput()

#this section is going to do a boolean off all the surfaces

# exclude theses surfaces
surface_exlusion_list = [11, 17]
surface_list = [i for i in region_list if (i not in surface_exlusion_list)]


surface_0_id = surface_list.pop(3)
print(surface_0_id, surface_list )
surface_0 = mesh_files[surface_0_id]
#surface_list = []
while(len(surface_list) > 0):
    for i in surface_list:
        cur_surf = i
        intersect_result = 0
        bool_result = 0
        try:
            intersectionPolyDataFilter = vtk.vtkIntersectionPolyDataFilter()
            intersectionPolyDataFilter.SetInputData( 0, surface_0)
            intersectionPolyDataFilter.SetInputData( 1, mesh_files[i] )
            intersectionPolyDataFilter.Update()
            intersect_result = intersectionPolyDataFilter.GetStatus()
            print("intersection status: {0}".format(intersect_result))
        except Exception as inst:
            print(type(inst))    # the exception instance
            #print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
            #print(" not intersecting")
            continue
        else:
            print("\nDid the intersection work?")
        #if(intersect_result == 1):
            #writeinter = vtk.vtkXMLPolyDataWriter()
            #writeinter.SetInputConnection(normals.GetOutputPort())
            #file_name = os.path.join(path_str[0], "{0}_{1}_intersect.vtp".format(split_ext[0], i))
            #writeinter.SetFileName(file_name)
            #writeinter.Write()
            #distance = vtk.vtkDistancePolyDataFilter()
            #distance.SetInputConnection( 0, intersection.GetOutputPort( 1 ) )
            #distance.SetInputConnection( 1, intersection.GetOutputPort( 2 ) )

        try:
            booleanOperationFilter = vtk.vtkLoopBooleanPolyDataFilter()
            booleanOperationFilter.SetInputConnection(intersection.GetOutputPort())
            #booleanOperationFilter.SetInputConnection(1, intersection.GetOutputPort( 2 ))
            booleanOperationFilter.SetOperationToUnion()
            booleanOperationFilter.SetTolerance(1.0e-7)
            booleanOperationFilter.Update()
            bool_result = booleanOperationFilter.GetStatus()
            print("boolean status: {0}".format(bool_result))
        except Exception as inst:
            print(type(inst))    # the exception instance
            #print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
            print("boolean fail")
            continue
        if(bool_result == 1):
            surface_0 = booleanOperationFilter.GetOutput()
            break
    print(cur_surf, len(surface_list))
    surface_0_id = surface_list.remove(cur_surf)

writer3 = vtk.vtkXMLPolyDataWriter()
writer3.SetInputData(surface_0)
file_name = os.path.join(path_str[0], "{0}_merged.vtp".format(split_ext[0]))
writer3.SetFileName(file_name)
writer3.Write()
