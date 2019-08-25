#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import itertools
import os

# clip the flow extensions

# this one is more accurate
# http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
def hughes_moeller (n):
# Choose a vector orthogonal to n as the direction of b2.
    if( np.fabs(n[0]) > np.fabs (n[2] )):
        b2 = np.array([-n[1], n[0] , 0.0])
    else:
        b2 = np.array([0.0, -n[2], n[1]])
    b2 *= (np.dot (b2 , b2 ))**(-0.5) # Normalize b2
    b1 = np.cross (b2 , n ) # Construct b1 using a cross product
    return b1, b2


def Execute(args):
    print("clip surface")
    
    mesh_reader = vmtkscripts.vmtkMeshReader()
    mesh_reader.InputFileName = args.mesh_file
    mesh_reader.Execute()
    
    mesh2surf = vmtkscripts.vmtkMeshToSurface()
    mesh2surf.Mesh = mesh_reader.Mesh
    mesh2surf.CleanOutput = 0
    mesh2surf.Execute()
    
    scale_cfd = vmtkscripts.vmtkSurfaceScaling()
    scale_cfd.ScaleFactor = args.scale # meters to mm
    scale_cfd.Surface = mesh2surf.Surface
    scale_cfd.Execute()
    
    surface = vtk.vtkPolyData()
    surface.DeepCopy(scale_cfd.Surface)
    
    reader_trim = vmtkscripts.vmtkSurfaceReader()
    reader_trim.InputFileName = args.polydata_trim
    reader_trim.Execute()
    br_trim = reader_trim.Surface
    
    reader_ext = vmtkscripts.vmtkSurfaceReader()
    reader_ext.InputFileName = args.polydata_ext
    reader_ext.Execute()
    br_ext = reader_ext.Surface
    
    # have to make sure that they both have the same number of GetNumberOfPoints
    assert br_trim.GetNumberOfPoints() == br_ext.GetNumberOfPoints()
    
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(br_trim)
    locator.BuildLocator()
    
    point_ext = [0.0, 0.0, 0.0]
    pt_cross = [0.0, 0.0, 0.0]
    pt_dot = 0.0
    count = 0
    for trim_id in range(br_ext.GetNumberOfPoints()):
        
        # get extension point
        point_ext = br_ext.GetPoint(trim_id)
        #closest trim point
        point_trim_id = locator.FindClosestPoint(point_ext)
        point_trim = br_trim.GetPoint(point_trim_id)

        # check that the points are close to the same direction
        pt_trim_normal = br_trim.GetPointData().GetArray("BoundaryNormals").GetTuple(point_trim_id)
        pt_ext_normal = br_ext.GetPointData().GetArray("BoundaryNormals").GetTuple(trim_id)
        
        #print(pt_trim_normal, pt_ext_normal)
        pt_dot = vtk.vtkMath.Dot(pt_trim_normal, pt_ext_normal)
        #vtk.vtkMath.Cross(pt_trim_normal, pt_ext_normal, pt_cross)
    
        #print(pt_dot, vtk.vtkMath.Norm(pt_cross))#, pt_cross)
        
        if ( pt_dot < 0.95):
            print("help the vectors aren't colinear")
            assert pt_dot > .95
            
        v =  np.array(point_ext) - np.array(point_trim) #pt1 - pt2
        v_mag = np.linalg.norm(v)
        n = v / v_mag
       # print("should be 1.0", np.linalg.norm(n), n)

        b1, b2 = hughes_moeller(n) #orthogonal basis


        #Get  maximum radius
        box_radius = br_ext.GetPointData().GetArray("BoundaryRadius").GetTuple(trim_id)
        box_radius_trim = br_trim.GetPointData().GetArray("BoundaryRadius").GetTuple(point_trim_id)
        #print(box_radius_trim, box_radius)
        
        extra_room = args.margin
        extra_z = 0.0
        r_max = extra_room*max([box_radius[0], box_radius_trim[0]])  # max radius
        z_max = extra_room*v_mag

        #create transformation matrix
        R = np.zeros((4,4), dtype=np.float64)
        R[:3,0] = b1 #x
        R[:3,1] = b2 #y
        R[:3,2] = n #z
        R[:3,3] = np.array(point_trim) # the beginning of the clip
        R[3,3] = 1.0

        trans_matrix = vtk.vtkTransform()
        trans_inverse = vtk.vtkTransform()

        trans_matrix.SetMatrix(list(R.ravel()))
        #print(trans_matrix.GetMatrix())

        trans_inverse.DeepCopy(trans_matrix)
        trans_inverse.Inverse()
        
        # point to define bounds
        dims_min = [-r_max, -r_max, -extra_z*z_max]
        dims_max = [r_max, r_max, z_max]
        
        planes = vtk.vtkBox()
        planes.SetBounds (dims_min[0], dims_max[0], dims_min[1], dims_max[1], dims_min[2], dims_max[2])
        planes.SetTransform(trans_inverse)
        
        clipper = vtk.vtkTableBasedClipDataSet()
        clipper.SetInputData(surface)
        clipper.SetClipFunction(planes)
        clipper.InsideOutOff()
        #clipper.SetMergeTolerance(1.0E-6)
        clipper.Update()
        #print(clipper.GetMergeTolerance())
        surface = clipper.GetOutput()

        #test = vtk.vtkCubeSource()
        #test.SetBounds (dims_min[0], dims_max[0], dims_min[1], dims_max[1], dims_min[2], dims_max[2])
        
        #trans_cube = vtk.vtkTransformPolyDataFilter()
        #trans_cube.SetInputConnection(test.GetOutputPort())
        #trans_cube.SetTransform(trans_matrix)
        #trans_cube.Update()
        
        #writer2 = vmtkscripts.vmtkSurfaceWriter()
        #writer2.OutputFileName = os.path.join(os.path.split(args.out_file)[0], "test_clip_box_{0}.vtp".format(count))
        #writer2.Input = trans_cube.GetOutput()
        #writer2.Execute()
        
        count += 1

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(surface)
    geom.Update()
    
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.OutputFileName = args.out_file
    writer.Input = geom.GetOutput()
    writer.Execute()




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Clip flow extensions')
    parser.add_argument("-i", dest="mesh_file", required=True, help="vtu CFD Mesh file to trim", metavar="FILE")
    parser.add_argument("--original", dest="polydata_trim", required=True, help="boundary reference trim", metavar="FILE")
    parser.add_argument("--extensions", dest="polydata_ext", required=True, help="boundary reference extensions", metavar="FILE")
    parser.add_argument("-o", dest="out_file", required=True,
                        help="output filename for clipped surface", metavar="FILE",
                        default = "case_wall_clip")
    parser.add_argument("--margin", dest="margin", type=float, help="specify the global  margin of the clip box", metavar="FLOAT",
                        default = 1.1)
    parser.add_argument("--scale", dest="scale", type=float, help="specify the mesh scale", metavar="FLOAT",
                        default = 1000.0)
    args = parser.parse_args()
    #print(args)
    Execute(args)




