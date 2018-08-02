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
    
    
    bifurcation_info = {}
    for i in range(centerlines.GetNumberOfCells()):
        bifurcation_info[i] = {"clip_id": [], "cell_id": []}
        cell = centerlines.GetCell(i)
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
             continue

        n_cell_pts = cell.GetNumberOfPoints()
        
        start_end_pt = [0, n_cell_pts-1]
            
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
                        if(size_ratio > 2.0  ):
                            continue
                        
                        cv, offset, shape = ComputeBranchSectionShape(cutPoly, point)
                        
                        if(cv > 0.2):
                            continue
                        if(offset > 0.10):
                            continue
                        #if(shape > 0.8):
                         #   continue
                        
                        
                        
                        #else:
                            #average area
                            #radius2 = max(radius, np.sqrt(area_test/np.pi))
                        
                        #shape = ComputeBranchSectionShape(cutPoly, point, size_range)
                        
                        writerline = vmtkscripts.vmtkSurfaceWriter()
                        writerline.OutputFileName = "test_loop_{0}.vtp".format(pd_count)
                        writerline.Input = cutPoly #ex.GetOutput()
                        writerline.Execute()
                        pd_count += 1
                        

                        #if (radius2 <= 0.0):
                            #radius2 = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius").GetTuple(cell.GetPointIds().GetId(k))[0]
                            ##if ( radius2 > radius):
                                ##radius = radius2
                            ##else:
                                ##ratio = radius/radius2
                                
                        #else:
                            
                        #print(length)
                        clip_id = cell.GetPointIds().GetId(k)
                        bifurcation_info[i]["clip_id"].append(clip_id)
                        bifurcation_info[i]["cell_id"].append(k)
                        break
        #return

    #t = [ 1 for i in bifurcation_info.keys() if len(bifurcation_info[i]) == 2]
    two_bif = False
    pd_count = 0
    for cell in bifurcation_info:
        id_sorted = sorted(bifurcation_info[cell]["cell_id"])
        
        if (len(bifurcation_info[cell]["cell_id"]) < 2):
            two_bif = False
        else:
            two_bif = True
            diff = bifurcation_info[cell]["cell_id"][0] - bifurcation_info[cell]["cell_id"][1]
            if(abs(diff) < 2): # there is a problem if there less than two points
                print("houston we got a problem")

        clip_id = centerlines.GetCell(cell).GetPointIds().GetId(id_sorted[0])
        clip_id_m1 = centerlines.GetCell(cell).GetPointIds().GetId(id_sorted[0]+1)
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
        
        #tree = vtk.vtkModifiedBSPTree()
        #tree.SetDataSet(surface_reference)
        #tree.BuildLocator()
        ##intersect the locator with the line
        #LineP0 = start_pt

        ## 200 points
        #radius_est = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius").GetTuple(clip_id)[0]
        ##radii increment is proportional to circumference
        ##distance between points
        #cnt_dist = 0.05
        #n_radii = int(np.pi*2.0*radius_est/cnt_dist)
        #dt = radius_est*4.0*cnt_dist #estimate ray step from radius
        #dtheta = [0.0 + i*(359.0-0.0)/(n_radii-1) for i in range(n_radii)] #[0.0]
        #out_vector = (0.0,0.0,0.0)
        #tolerance = 0.0000001

        #polylines = vtk.vtkCellArray()
        #cut_surface = vtk.vtkPolyData()
        #new_line = vtk.vtkPolyLine()
        #new_line.GetPointIds().SetNumberOfIds(len(dtheta)+1)
        
        
        #IntersectPointsList = vtk.vtkPoints()
        #loop_pts_list = vtk.vtkPoints()
        #IntersectCellsList = vtk.vtkIdList()
        #for idx, theta in enumerate(dtheta):
            #IntersectPoints = vtk.vtkPoints()
            #IntersectCells = vtk.vtkIdList()
            #code = 0
            #count = 1
            #rotate = vtk.vtkTransform()
            #rotate.RotateWXYZ(theta,  v_ctr_start)
            #rotate.Update()

            ##print(dir(rotate))
            ##trans_m = vtk.vtkMatrix4x4()
            ##rotate.GetMatrix(trans_m)

            #out_vector = rotate.TransformVector(v_normal_start)
            #LineP1 = [ c2 + count*dt*c1 for c2, c1 in zip(start_pt, out_vector)]
            ##print(v_normal_start, out_vector)
            #while ( code == 0 and count < 10000):
                #count += 1
                #code = tree.IntersectWithLine(LineP0, LineP1,
                                                #tolerance, IntersectPoints,
                                                #IntersectCells)
                #LineP1 = [ c2 + count*dt*c1 for c2, c1 in zip(start_pt, out_vector)]
            #if(count > 10000 and code == 0):
                #print("no intersection")
                #continue

            #if (code != 0):
                #if(IntersectCells.GetNumberOfIds() > 1):
                    #print(IntersectCells.GetNumberOfIds())
                #pt = IntersectPoints.GetPoint(0)
                ##pt = [ c2 + dt*c1 for c2, c1 in zip(pt, out_vector)] # add some buffer, may not need it
                #new_pt_id = IntersectPointsList.InsertNextPoint(pt)
                #new_line.GetPointIds().SetId(idx, new_pt_id)
                #loop_pts_list.InsertNextPoint(LineP1)
                #IntersectCellsList.InsertNextId(IntersectCells.GetId(0))
                ##print(IntersectPoints.GetPoint(0), IntersectCells.GetId(0) )
        
        #new_line.GetPointIds().SetId(len(dtheta), 0)
        #print(IntersectPointsList.GetPoint(0))
        #print(v_ctr_start, start_pt)
        #polylines.InsertNextCell(new_line)
        #cut_surface.SetPoints(IntersectPointsList)
        #cut_surface.SetLines(polylines)

        #writerline = vmtkscripts.vmtkSurfaceWriter()
        #writerline.OutputFileName = "test_loop_{0}.vtp".format(pd_count)

        #writerline.Input = cut_surface
        #writerline.Execute()


        cutter = vtk.vtkCutter()
        cutter.SetInputData(surface_reference)
        cutter.SetCutFunction(plane1)
        cutter.Update()


        extract = vtk.vtkPolyDataConnectivityFilter()
        extract.SetInputConnection(cutter.GetOutputPort())
        extract.SetExtractionModeToClosestPointRegion()
        extract.SetClosestPoint(start_pt)

        extract.Update()

        loop = extract.GetOutput()
        weights = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        gencell = vtk.vtkGenericCell()
        cross_inter = [0.0,0.0,0.0]
        cross_edges = [0.0,0.0,0.0]
        cross_test = [0.0,0.0,0.0]
        test_pt = [0.0,0.0,0.0]
        thresh = 0.0
        first_3tris = False
        for i in range(loop.GetNumberOfCells()):
            pt1 = loop.GetPoint(loop.GetCell(i).GetPointIds().GetId(0))
            pt2 = loop.GetPoint(loop.GetCell(i).GetPointIds().GetId(1))
            mid_pt = [ (x+y)/2.0 for x,y in zip(pt2,pt1)]
            direction = [ x-y for x,y in zip(pt2,pt1)]
            
            cell_id = locator_cell.FindCell(mid_pt, 0.0001, gencell, test_pt, weights)
            
            cell_ = surface_reference.GetCell(cell_id)
            
            right = []
            left = []
            center = []
            pt_list = []
            for j in range(cell_.GetNumberOfPoints()):
                #get distance
                pt_list.append(surface_reference.GetPoint(cell_.GetPointIds().GetId(j)))
                dist = plane1.EvaluateFunction(pt_list[-1])
                if ( dist < -thresh):
                    left.append(j)
                elif (dist > thresh):
                    right.append(j)
                else:
                    center.append(j)
            
            tag = ""
            if len(center) > 1:
                # don't do anything its already split on edge
                tag = "edge"
                print("edge")
            elif len(center) > 0:
                # split into two triangles
                pt = center[0]
                tag = "2_tris"
            else:
                tag = "3_tris"
            
                if (len(left) > 1):
                    #print("left")
                    pt = right[0]
                elif (len(right) > 1): 
                    pt = left[0]
                else:
                    print("split triangle")
            
            edge1 = [ x-y for x,y in zip(pt_list[(pt+1)%3],pt_list[pt])]
            edge2 = [ x-y for x,y in zip(pt_list[(pt+2)%3],pt_list[pt])]
            
            vtk.vtkMath.Cross(edge1, edge2, cross_edges)
            vtk.vtkMath.Normalize(cross_edges)
            vtk.vtkMath.Cross(edge1, direction, cross_test)
            vtk.vtkMath.Normalize(cross_test)
            
            is_winding = vtk.vtkMath.Dot(cross_edges, cross_test)
            
            # switch the winding of the intersection points
            if(is_winding < 0.0):
                tmp = pt1
                pt1 = pt2
                pt2 = tmp
                
            if ( tag == "3_tris"):
                if(first_3tris == False):
                    first_3tris = True
                    
                # first triangle
                #new_cell = vtk.vtkTriangle()
                #pts_id_list = []
                #pt_id_1 = triangle_pts.InsertNextPoint(pt_list[pt])
                #new_cell.GetPointIds().SetId(0, pt_id_1)
                
                #pt_id_2 = triangle_pts.InsertNextPoint(pt1)
                #new_cell.GetPointIds().SetId(1, pt_id_2)
                
                #pt_id_3 = triangle_pts.InsertNextPoint(pt2)
                #new_cell.GetPointIds().SetId(2, pt_id_3)
                
                #triangles.InsertNextCell(new_cell)
                
                triangle_pts = vtk.vtkPoints()
                
                quad_id_1 = triangle_pts.InsertNextPoint(pt2)
                quad_id_2 = triangle_pts.InsertNextPoint(pt1)
                quad_id_3 = triangle_pts.InsertNextPoint(pt_list[(pt+1)%3])
                quad_id_4 = triangle_pts.InsertNextPoint(pt_list[(pt+2)%3])
                
                pts_new_triangle = []
                
                pt_id_2 = surface_reference.GetPoints().InsertNextPoint(pt1)
                pts_new_triangle.append(surface_reference.GetCell(cell_id).GetPointIds().GetId(pt))
                pts_new_triangle.append(pt_id_2)
                surface_reference.GetPointData().GetArray("Ids").InsertNextTuple([pt_id_2])
                pt_id_2_old = surface_reference.GetCell(cell_id).GetPointIds().GetId((pt+1)%3)
                #surface_reference.GetCell(cell_id).GetPointIds().SetId((pt+1)%3, pt_id_2)
                
                pt_id_3 = surface_reference.GetPoints().InsertNextPoint(pt2)
                pts_new_triangle.append(pt_id_3)
                surface_reference.GetPointData().GetArray("Ids").InsertNextTuple([pt_id_2])
                pt_id_3_old = surface_reference.GetCell(cell_id).GetPointIds().GetId((pt+2)%3)
                #surface_reference.GetCell(cell_id).GetPointIds().SetId((pt+2)%3, pt_id_3)

                surface_reference.ReplaceCell(cell_id, len(pts_new_triangle), pts_new_triangle)
                # map polygon to reference mesh
                map_to = {quad_id_1 : pt_id_3, quad_id_2 :  pt_id_2, quad_id_3 : pt_id_2_old, quad_id_4 : pt_id_3_old}
                
                npts = 4
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(npts)
                polygon.GetPoints().SetNumberOfPoints(npts)
                
                polygon.GetPointIds().SetId(0, quad_id_1)
                polygon.GetPoints().SetPoint(0, triangle_pts.GetPoint(quad_id_1))
                polygon.GetPointIds().SetId(1, quad_id_2)
                polygon.GetPoints().SetPoint(1, triangle_pts.GetPoint(quad_id_2))
                
                polygon.GetPointIds().SetId(2, quad_id_3)
                polygon.GetPoints().SetPoint(2, triangle_pts.GetPoint(quad_id_3))
                polygon.GetPointIds().SetId(3, quad_id_4)
                polygon.GetPoints().SetPoint(3, triangle_pts.GetPoint(quad_id_4))
                
                quad_ids = vtk.vtkIdList()
                
                polygon.Triangulate(quad_ids)
                numPts = quad_ids.GetNumberOfIds()
                numSimplices = numPts // 3
                triPts = [0,0,0]
                triPts_map = [0,0,0]
                #print(numSimplices, numPts
                for  j in range(numSimplices):
                    for k in range(3):
                        triPts[k] = polygon.GetPointIds().GetId(quad_ids.GetId(int(3*j+k)))
                        triPts_map[k] = map_to[triPts[k]]
                    #triangles.InsertNextCell(3, triPts)
                    
                    cell_id_new = surface_reference.GetPolys().InsertNextCell(3,triPts_map)
                    surface_reference.GetCellData().GetArray("Ids").InsertNextTuple([cell_id_new])
                #surface_reference.Modified()
                #print("hello")
            

            #if ( tag == "2_tris"):
                ## doesnt' work well
                ## for collapse of intersection line on
                #new_cell = vtk.vtkTriangle()
                #pts_id_list = []
                #pt_id_1 = triangle_pts.InsertNextPoint(pt_list[pt])
                #new_cell.GetPointIds().SetId(0, pt_id_1)
                
                #pt_id_2 = triangle_pts.InsertNextPoint(pt_list[(pt+1)%3])
                #new_cell.GetPointIds().SetId(1, pt_id_2)
                
                #pt_id_3 = triangle_pts.InsertNextPoint(pt2)
                #new_cell.GetPointIds().SetId(2, pt_id_3)
                
                #triangles.InsertNextCell(new_cell)
                
                #new_cell = vtk.vtkTriangle()
                #new_cell.GetPointIds().SetId(0, pt_id_1)
                #new_cell.GetPointIds().SetId(1, pt_id_3)
                #pt_id_4 = triangle_pts.InsertNextPoint(pt_list[(pt+2)%3])
                #new_cell.GetPointIds().SetId(2, pt_id_4)

                #triangles.InsertNextCell(new_cell)
                
        
        
        #triangle_pd.SetPoints(triangle_pts)
        #triangle_pd.SetPolys(triangles)
        
        #pass_ = vtk.vtkPassArrays()
        #pass_.SetInputData(surface_reference)
        #pass_.RemoveArraysOn()
        #pass_.RemoveCellDataArray("Ids")
        #pass_.RemoveCellDataArray("Normals")
        #pass_.RemovePointDataArray("Ids")
        #pass_.RemovePointDataArray("Normals")
        
        ##pass_.ClearPointDataArrays()
        ##pass_.ClearCellDataArrays()
        #pass_.Update()
        
        #geom = vtk.vtkGeometryFilter()
        #geom.SetInputConnection(pass_.GetOutputPort())
        #geom.Update()
        
        #normalsFilter2 = vmtkscripts.vmtkSurfaceNormals()
        #normalsFilter2.ComputeCellNormals = 1
        #normalsFilter2.Surface = surface_reference
        #normalsFilter2.NormalsArrayName = 'Normals'
        #normalsFilter2.Execute()
        
        #writer = vmtkscripts.vmtkSurfaceWriter()
        #writer.OutputFileName = "test_file_{0}.vtp".format(pd_count)
        #writer.Input = surface_reference #geom.GetOutput() #triangle_pd #extract.GetOutput()
        #writer.Execute()
        pd_count += 1
        
        #print("yp")
        surface_reference.Modified()
        #print("zzz")
        surface_reference.BuildCells()
        surface_reference.BuildLinks()
        #print("yppp")
        locator_surf = vtk.vtkPointLocator()
        locator_surf.SetDataSet(surface_reference)
        locator_surf.BuildLocator()
        
        #print("ydddp")
        locator_cell = vtk.vtkCellLocator()
        locator_cell.SetDataSet(surface_reference)
        locator_cell.BuildLocator()
        
        

        #return

    print( bifurcation_info)
    #return
    
    normalsFilter2 = vmtkscripts.vmtkSurfaceNormals()
    normalsFilter2.ComputeCellNormals = 1
    normalsFilter2.Surface = surface_reference
    normalsFilter2.NormalsArrayName = 'Normals'
    normalsFilter2.Execute()
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file
    writer.Input = normalsFilter2.Surface
    writer.Execute()
    
            
        #length = 0.0
        #start_pt_idx = 0 
        #for j in range(n_cell_pts):
            #centerlines.GetPointCells(cell.GetPointIds().GetId(j), cell_Ids)
            #n_pt_neighbors = cell_Ids.GetNumberOfIds()

            #pt_id = cell.GetPointIds().GetId(j)
            #pts_ids.InsertNextId(pt_id)
            #point = centerlines.GetPoint(cell.GetPointIds().GetId(j))
            #length += vtk.vtkMath.Distance2BetweenPoints(prevPoint,point)**0.5
            #prevPoint = point
            #if((j > start_pt_idx  and n_pt_neighbors > 1) or  (j == n_cell_pts-1)):
                ##close

                #new_polyline = addPolyLine(pts_ids)
                ## weird issue with duplicate points if they are not removed
                #if(length > 0.0):
                    #outputLines.InsertNextCell(new_polyline)
                    #lengthArray.InsertNextTuple([length])
                #start_pt_idx = j
                #if(n_pt_neighbors > 1):
                    #pts_ids.Reset()
                    #pts_ids.InsertNextId(pt_id)
                    #length = 0.0
        #pts_ids.Reset()

    #output.SetPoints(centerlines.GetPoints())
    #output.SetLines(outputLines)
    #output.GetCellData().AddArray(lengthArray)
    #for i in range(centerlines.GetPointData().GetNumberOfArrays()):
        #output.GetPointData().AddArray(centerlines.GetPointData().GetArray(i))
    
    
    #writer = vmtkscripts.vmtkSurfaceWriter()
    #writer.OutputFileName = args.out_file

    #if(args.clean_ctr):
        #cleaner2 = vtk.vtkCleanPolyData()
        #cleaner2.PointMergingOn()
        #cleaner2.SetInputData(output)
        #cleaner2.Update()
        #writer.Input = cleaner2.GetOutput()
    #else:
        #writer.Input = output

    #writer.Execute()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface", required=True, help="input surface file", metavar="FILE")
    parser.add_argument("-c", dest="centerlines", required=True, help="centerlines", metavar="FILE")
    parser.add_argument("--clean", dest="clean_ctr", action='store_true', help=" clean centerlines after")
    parser.add_argument("-o", dest="out_file", required=True, help="output filename for labeled surface mesh", metavar="FILE")
    parser.add_argument("-s", dest="out_segments", required=True, help="output filename for evaluated centerlines and surface mesh (slices)", metavar="FILE")
    args = parser.parse_args()
    #print(args)
    Execute(args)




