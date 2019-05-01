#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
from vmtk import vtkvmtk
import argparse
import itertools
import os

def close_cell(section):
    #assume the cell array of lines
    section.BuildCells()
    section.BuildLinks()
    
    numberOfLinePoints = section.GetNumberOfPoints()
    
    cell_ids = vtk.vtkIdList()
    
    numberOfSingleCellPoints = 0
    
    termination_pts = []
    
    for i in range(section.GetNumberOfPoints()):
        section.GetPointCells(i,cell_ids)
        if(cell_ids.GetNumberOfIds() == 1):
            numberOfSingleCellPoints += 1
            termination_pts.append(i)
            
    
    
    if(numberOfSingleCellPoints == 2):
        print(termination_pts)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, termination_pts[0])
        line.GetPointIds().SetId(1, termination_pts[1])
    
        section.GetLines().InsertNextCell(line)
    elif(numberOfSingleCellPoints > 2):
        print("disconnected section")

def ComputePolygonArea(section):
    # calculate area of closed polygon
    section.BuildCells()
    
    section_area = 0.0
    area_calc = 0.0
    
    if (section.GetNumberOfCells() == 0):
        print("shwarma")
        return section_area
    elif (section.GetNumberOfCells() > 1):
        print("there should only be one cell")

    trianglePointIds = vtk.vtkIdList()
    points_list = vtk.vtkPoints()
    for j in range(section.GetNumberOfCells()):
        area_calc = 0.0
        cell = section.GetCell(j)
        if ( cell.GetCellType() != vtk.VTK_POLYGON ):
            print(cell.GetCellType())
            continue
            #cell.Triangulate(j, trianglePointIds, points_list)
        
        cell.Triangulate(trianglePointIds)
    
        numberOfTriangles = trianglePointIds.GetNumberOfIds() // 3
        #print("triangles", numberOfTriangles)
        point0 = [0.0,0.0,0.0]
        point1 = [0.0,0.0,0.0]
        point2 = [0.0,0.0,0.0]
        
        for  i in range(numberOfTriangles):
            pointId0 = trianglePointIds.GetId(3*i)
            pointId1 = trianglePointIds.GetId(3*i+1)
            pointId2 = trianglePointIds.GetId(3*i+2)

            cell.GetPoints().GetPoint(pointId0, point0)
            cell.GetPoints().GetPoint(pointId1, point1)
            cell.GetPoints().GetPoint(pointId2, point2)

            area_calc += vtk.vtkTriangle.TriangleArea(point0,point1,point2)
        
        section_area = max(area_calc, section_area)
    
    return section_area



def ComputeBranchSectionShape(section, origin):
    # eccentricity of slice
    pointIds = vtk.vtkIdList()
    #cv = 0.0
    #offset = 0.0
    #shape = 0.0
    for j in range(section.GetNumberOfCells()):
        area_calc = 0.0
        cell = section.GetCell(j)
        if ( cell.GetCellType() != vtk.VTK_POLYGON ):
            print(cell.GetCellType())
            continue
        center = [0.0,0.0,0.0]
        for i in range(cell.GetNumberOfPoints()):
            pt = section.GetPoint(cell.GetPointIds().GetId(i))
            center = [p+c for p,c in zip(pt, center)]
        center = [p/cell.GetNumberOfPoints() for p in center]
        
        diff_origin = (vtk.vtkMath.Distance2BetweenPoints(center, origin))**0.5
        
        rad_list = []
        for i in range(cell.GetNumberOfPoints()):
            pt = section.GetPoint(cell.GetPointIds().GetId(i))
            radius = (vtk.vtkMath.Distance2BetweenPoints(origin, pt))**0.5
            rad_list.append(radius)

        mean = np.mean(rad_list)
        stddev = np.std(rad_list)
        
        shape = min(rad_list)/max(rad_list)
        
        cv = (1.0 + 1.0/(4.0*cell.GetNumberOfPoints()))*stddev/mean
        offset = diff_origin/mean


        #print(mean, stddev, cv, offset, shape)

    return cv, offset, shape
 

# get the average radius for each segment

def Execute(args):
    print("evaluate centerlines")
    
    reader_ctr = vmtkscripts.vmtkSurfaceReader()
    reader_ctr.InputFileName = args.centerlines
    reader_ctr.Execute()
    
    print(args.clean_ctr)
    if(args.clean_ctr):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.PointMergingOn()
        cleaner.SetInputData(reader_ctr.Surface)
        cleaner.Update()
        centerlines = cleaner.GetOutput()
    else:
        centerlines = reader_ctr.Surface
    
    centerlines.BuildLinks()
    centerlines.BuildCells()
    
    # calculate length for each segment
    # seems to be some error in prevous calculation
    for i in range(centerlines.GetNumberOfCells()):
        cell = centerlines.GetCell(i)
        length_ = 0.0
        prevPoint = cell.GetPoints().GetPoint(0)
        for j in range(cell.GetNumberOfPoints()):
            point = cell.GetPoints().GetPoint(j)
            length_ += vtk.vtkMath.Distance2BetweenPoints(prevPoint,point)**0.5
            prevPoint = point
        centerlines.GetCellData().GetArray("length").SetTuple(i, [length_])
        
    #writer2 = vmtkscripts.vmtkSurfaceWriter()
    #writer2.OutputFileName = "centerlines_test.vtp"
    #writer2.Input = centerlines
    #writer2.Execute()
    
    
    reader_br = vmtkscripts.vmtkSurfaceReader()
    reader_br.InputFileName = args.surface
    reader_br.Execute()
    
    #if (reader_br.Surface.GetPointData().GetNormals() == None):
        #normalsFilter = vmtkscripts.vmtkSurfaceNormals()
        #normalsFilter.ComputeCellNormals = 1
        #normalsFilter.Surface = reader_br.Surface
        #normalsFilter.NormalsArrayName = 'Normals'
        #normalsFilter.Execute()
        #surface_reference = normalsFilter.Surface
    #else:
    surface_reference = reader_br.Surface
        
    
    locator_surf = vtk.vtkPointLocator()
    locator_surf.SetDataSet(surface_reference)
    locator_surf.BuildLocator()
    
    locator_cell = vtk.vtkCellLocator()
    locator_cell.SetDataSet(surface_reference)
    locator_cell.BuildLocator()

    
    cell_Ids = vtk.vtkIdList()
    outputLines = vtk.vtkCellArray()
    output = vtk.vtkPolyData()
    
    triangles = vtk.vtkCellArray()
    triangle_pd = vtk.vtkPolyData()
    triangle_pts = vtk.vtkPoints()
    
    lengthArray = vtk.vtkDoubleArray()
    lengthArray.SetName("length")
    lengthArray.SetNumberOfComponents(1)
    
    pts_ids = vtk.vtkIdList()
    factor = 1.0
    factor2 = 2.0
    pd_count = 0
    
    size_range = [0.0, 0.0]
    
    mid_points = vtk.vtkPoints()
    vertex = vtk.vtkCellArray()
    
    bifurcation_info = {}
    for i in range(centerlines.GetNumberOfCells()):
        bifurcation_info[i] = {"clip_id": [], "cell_pt_id": [], "mid_pt": [], "step":[], "less_length": 0.0}
        cell = centerlines.GetCell(i)
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
             continue

        n_cell_pts = cell.GetNumberOfPoints()
        
        start_end_pt = [0, n_cell_pts-1]
        cell_length_half = centerlines.GetCellData().GetArray("length").GetTuple(i)[0]/2.0
        
        for j in start_end_pt:
            pt_id_pd = cell.GetPointIds().GetId(j)
            
            centerlines.GetPointCells(pt_id_pd, cell_Ids)
            if (cell_Ids.GetNumberOfIds() > 1):
                radius = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius").GetTuple(pt_id_pd)[0]
                length = 0.0
                radius2 = 0.0
                prev_point = centerlines.GetPoint(pt_id_pd)
                if( j == start_end_pt[0]):
                    step = 1
                    stop = start_end_pt[-1]
                else:
                    step = -1
                    stop = -1
                for k in range(j, stop, step):
                    point = centerlines.GetPoint(cell.GetPointIds().GetId(k))
                    length += vtk.vtkMath.Distance2BetweenPoints(prev_point,point)**0.5
                    prev_point = point
                    if (length > (factor*radius + factor2*radius2)):
                        #print(length)
                        pl_vec = centerlines.GetPointData().GetArray("FrenetTangent").GetTuple(cell.GetPointIds().GetId(k))
                        pl = vtk.vtkPlane()
                        pl.SetOrigin(point)
                        pl.SetNormal(pl_vec)
        
                        cut = vtk.vtkCutter()
                        cut.SetInputData(surface_reference)
                        cut.SetCutFunction(pl)
                        cut.Update()

                        ex = vtk.vtkPolyDataConnectivityFilter()
                        ex.SetInputConnection(cut.GetOutputPort())
                        #ex.SetExtractionModeToAllRegions()
                        ex.SetExtractionModeToClosestPointRegion()
                        ex.SetClosestPoint(point)
                        ex.Update()
                        
                        lp = ex.GetOutput()
                        close_cell(lp)

                        cutStrips = vtk.vtkStripper()  # Forms loops (closed polylines) from cutter
                        cutStrips.SetInputData(lp)
                        cutStrips.Update()
                        cutPoly = vtk.vtkPolyData()  # This trick defines polygons as polyline loop
                        cutPoly.SetPoints((cutStrips.GetOutput()).GetPoints())
                        cutPoly.SetPolys((cutStrips.GetOutput()).GetLines())
                        
                        area_test = ComputePolygonArea(cutPoly)
                        size_ratio = area_test/(np.pi*radius**2)
                        #print(area_test, radius, size_ratio)
                        
                        #writerline = vmtkscripts.vmtkSurfaceWriter()
                        #writerline.OutputFileName = "test_loop_{0}.vtp".format(pd_count)
                        #writerline.Input = cutPoly #ex.GetOutput()
                        #writerline.Execute()
                        #pd_count += 1
                        if (length < cell_length_half):
                            if(size_ratio > 2.0  ):
                                continue
                            
                            cv, offset, shape = ComputeBranchSectionShape(cutPoly, point)
                            
                            if(cv > 0.2): # standard deviation / mean 
                                continue
                            if(offset > 0.10): # centroid of slice vs centerline point
                                continue
                        #if(shape > 0.8):
                         #   continue
                        
                        #writerline = vmtkscripts.vmtkSurfaceWriter()
                        #writerline.OutputFileName = "test_loop_{0}.vtp".format(pd_count)
                        #writerline.Input = cutPoly #ex.GetOutput()
                        #writerline.Execute()
                        #pd_count += 1
                            
                        #print(length)
                        clip_id = cell.GetPointIds().GetId(k)
                        bifurcation_info[i]["clip_id"].append(clip_id)
                        bifurcation_info[i]["cell_pt_id"].append(k)
                        bifurcation_info[i]["step"].append(step)
                        bifurcation_info[i]["less_length"] += length
                        tmp_idx = k 
                        break
                    
                midway_length = 0.0
                
                prev_point = centerlines.GetPoint(pt_id_pd)
                print("hello")
                for k in range(tmp_idx, stop, step):
                    if k == 1198:
                        print(k)
                    point = centerlines.GetPoint(cell.GetPointIds().GetId(k))
                    midway_length += vtk.vtkMath.Distance2BetweenPoints(prev_point, point)**0.5
                    prev_point = point
                    if (midway_length >= cell_length_half):
                        bifurcation_info[i]["mid_pt"].append(point)
                        pt_id = mid_points.InsertNextPoint(point)
                        vertex.InsertNextCell(1, [pt_id])
                        mid_idx = k
                        break

    
    mid_point_pd = vtk.vtkPolyData()
    mid_point_pd.SetPoints(mid_points)
    mid_point_pd.SetVerts(vertex)
    
    writerline = vmtkscripts.vmtkSurfaceWriter()
    writerline.OutputFileName = "test_vertex_{0}.vtp".format(0)

    writerline.Input = mid_point_pd
    writerline.Execute()
    
                    
        #return

    tree = vtk.vtkModifiedBSPTree()
    tree.SetDataSet(surface_reference)
    tree.BuildLocator()

    #t = [ 1 for i in bifurcation_info.keys() if len(bifurcation_info[i]) == 2]
    two_bif = False
    pd_count = 0

    avg_x_area = vtk.vtkDoubleArray()
    avg_x_area.SetName("avg_crosssection")
    avg_x_area.SetNumberOfComponents(1)
    avg_x_area.SetNumberOfTuples(centerlines.GetNumberOfCells())
    avg_x_area.Fill(-1.0)
    
    aspect_ratio = vtk.vtkDoubleArray()
    aspect_ratio.SetName("aspect_ratio")
    aspect_ratio.SetNumberOfComponents(1)
    aspect_ratio.SetNumberOfTuples(centerlines.GetNumberOfCells())
    aspect_ratio.Fill(-1.0)
    
    vol_array = vtk.vtkDoubleArray()
    vol_array.SetName("volume")
    vol_array.SetNumberOfComponents(1)
    vol_array.SetNumberOfTuples(centerlines.GetNumberOfCells())
    vol_array.Fill(-1.0)
    
    len_array = vtk.vtkDoubleArray()
    len_array.SetName("length_wo_bifurcation")
    len_array.SetNumberOfComponents(1)
    len_array.SetNumberOfTuples(centerlines.GetNumberOfCells())
    len_array.Fill(-1.0)
    
    append = vtk.vtkAppendPolyData()
    
    for cell_id in bifurcation_info:
        id_sorted = sorted(bifurcation_info[cell_id]["cell_pt_id"])
        step_direction = [x for _,x in sorted(zip(bifurcation_info[cell_id]["cell_pt_id"], bifurcation_info[cell_id]["step"]))]
        #print(step_direction)
        
        if (len(bifurcation_info[cell_id]["cell_pt_id"]) < 2):
            two_bif = False
        else:
            two_bif = True
            diff = bifurcation_info[cell_id]["cell_pt_id"][0] - bifurcation_info[cell_id]["cell_pt_id"][1]
            if(abs(diff) < 2): # there is a problem if there less than two points
                print("houston we got a problem")
                
        if (not two_bif):
            
            clip_id = centerlines.GetCell(cell_id).GetPointIds().GetId(id_sorted[0])
            clip_id_m1 = centerlines.GetCell(cell_id).GetPointIds().GetId(id_sorted[0]+step_direction[0])
            start_pt = centerlines.GetPoint(clip_id)
            surface_pt_id = locator_surf.FindClosestPoint(start_pt)
        
            # vector from pt(start_pt+1) - pt(start_pt) 
            v_start = [ x - y for x,y in zip(centerlines.GetPoint(clip_id_m1), start_pt)]
            v_ctr_start = centerlines.GetPointData().GetArray("FrenetTangent").GetTuple(clip_id)
            v_normal_start = centerlines.GetPointData().GetArray("FrenetNormal").GetTuple(clip_id)
            
       
            # want inward facing normals
            if (vtk.vtkMath.Dot(v_start, v_ctr_start) < 0.0):
                v_ctr_start = [-1.0*x for x in v_ctr_start]

            #print(clip_tangent)
            plane1 = vtk.vtkPlane()
            plane1.SetOrigin(start_pt)
            plane1.SetNormal(v_ctr_start)
            
            seamFilter = vtkvmtk.vtkvmtkTopologicalSeamFilter()
            seamFilter.SetInputData(surface_reference)
            seamFilter.SetClosestPoint(surface_reference.GetPoint(surface_pt_id))
            seamFilter.SetSeamScalarsArrayName("SeamScalars")
            seamFilter.SetSeamFunction(plane1)

            clipper = vtk.vtkClipPolyData()
            clipper.SetInputConnection(seamFilter.GetOutputPort())
            clipper.GenerateClipScalarsOff()
            clipper.GenerateClippedOutputOn()

            connectivity = vtk.vtkPolyDataConnectivityFilter()
            connectivity.SetInputConnection(clipper.GetOutputPort())
            connectivity.SetExtractionModeToClosestPointRegion()
            
            surface_mid_pt = locator_surf.FindClosestPoint(bifurcation_info[cell_id]["mid_pt"][0])
            connectivity.SetClosestPoint(surface_reference.GetPoint(surface_mid_pt))

            surfaceCleaner = vtk.vtkCleanPolyData()
            surfaceCleaner.SetInputConnection(connectivity.GetOutputPort())
            surfaceCleaner.Update()

            surfaceTriangulator = vtk.vtkTriangleFilter()
            surfaceTriangulator.SetInputConnection(surfaceCleaner.GetOutputPort())
            surfaceTriangulator.PassLinesOff()
            surfaceTriangulator.PassVertsOff()
            surfaceTriangulator.Update()
            
            capper = vmtkscripts.vmtkSurfaceCapper()
            capper.Surface = surfaceTriangulator.GetOutput()
            capper.Method = "simple"
            capper.Interactive = 0
            capper.Execute()
            
            get_prop = vtk.vtkMassProperties()
            get_prop.SetInputData(capper.Surface)
            get_prop.Update()
            
            volume = get_prop.GetVolume()
            new_length = centerlines.GetCellData().GetArray("length").GetTuple(cell_id)[0] - bifurcation_info[cell_id]["less_length"]
            average_area = volume/new_length
            
            avg_x_area.SetTuple(cell_id, [average_area])
            aspect_ratio.SetTuple(cell_id, [average_area/new_length])
            vol_array.SetTuple(cell_id, [volume])
            len_array.SetTuple(cell_id, [new_length])
            
            
            append.AddInputData(capper.Surface)
            append.Update()
            #print(new_length, centerlines.GetCellData().GetArray("length").GetTuple(cell_id)[0], bifurcation_info[cell_id]["less_length"])


            #pd_count += 1
    
    writerline = vmtkscripts.vmtkSurfaceWriter()
    writerline.OutputFileName = args.out_file
    writerline.Input = append.GetOutput()
    writerline.Execute()

    #print( bifurcation_info)
    centerlines.GetCellData().AddArray(avg_x_area)
    centerlines.GetCellData().AddArray(aspect_ratio)
    centerlines.GetCellData().AddArray(vol_array)
    centerlines.GetCellData().AddArray(len_array)
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_segments
    writer.Input = centerlines
    writer.Execute()
    
            



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    parser.add_argument("-c", dest="centerlines", required=True, help="centerlines", metavar="FILE")
    parser.add_argument("--clean", dest="clean_ctr", action='store_true', help=" clean centerlines after")
    parser.add_argument("-o", dest="out_file", required=False, help="output filename for terminal segments", metavar="FILE")
    parser.add_argument("-s", dest="out_segments", required=True, help="centerlines with cross section information", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)




