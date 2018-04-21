#!/usr/bin/env python

import vtk
import numpy as np
from vmtk import vmtkscripts
import argparse
import itertools
import os

# clip boxes along the surface

# calculate basis from vector
# Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani, and Ryusuke Villemin, Building an Orthonormal Basis, Revisited, Journal of Computer Graphics Techniques (JCGT), vol. 6, no. 1, 1-8, 2017
def revised_ONB(n):
    if (n[2] < 0.0):
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1 = [1.0 - n[0] * n[0] * a, -b, n[0]]
        b2 = [b, n[1] * n[1]*a - 1.0, -n[1]]
    else:
        a = 1.0 / (1.0 - n[2])
        b = -n[0] * n[1] * a
        b1 = [1.0 - n[0] * n[0] * a, b, -n[0]]
        b2 = [b, 1.0 - n[1] * n[1]*a, -n[1]]
    return b1, b2


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
    
    reader_ctr = vmtkscripts.vmtkSurfaceReader()
    reader_ctr.InputFileName = args.centerlines
    reader_ctr.Execute()
    centerlines = reader_ctr.Surface
    
    reader_surface = vmtkscripts.vmtkSurfaceReader()
    reader_surface.InputFileName = args.surface_file
    reader_surface.Execute()
    input_surface = reader_surface.Surface  
    
    dx = args.slice_thickness
    # only get the first three
    centroid = [float(i) for i in args.centroid.strip(" ").split(",")][0:3]
    neck_centroid = np.array(centroid)

    n_pts = centerlines.GetNumberOfPoints()
    pt1 = np.array(centerlines.GetPoint(0)) # start point
    pt2 = np.array(centerlines.GetPoint(n_pts-1)) # end point

    #print(pt1, pt2)
    v =  pt2 - pt1 #pt1 - pt2
    v_mag = np.linalg.norm(v)
    n = v / v_mag
    print("should be 1.0", np.linalg.norm(n), n)

    b1, b2 = hughes_moeller(n) #orthogonal basis


    #Get  maximum radius
    radius_range = [0.0, 0.0]
    centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius").GetRange(radius_range, 0)
    print(radius_range)

    r_max = 2*radius_range[-1] # max radius

    #https://en.wikipedia.org/wiki/Vector_projection
    # get starting point from centroid by projecting centroid onto normal direction
    neck_projection = np.dot(neck_centroid-pt1, n)*n
    neck_start_pt = pt1 + neck_projection
    print(neck_start_pt)

    #create transformation matrix
    R = np.zeros((4,4), dtype=np.float64)
    R[:3,0] = b1 #x
    R[:3,1] = b2 #y
    R[:3,2] = n #z
    R[:3,3] = neck_start_pt
    R[3,3] = 1.0

    trans_matrix = vtk.vtkTransform()
    trans_inverse = vtk.vtkTransform()

    trans_matrix.SetMatrix(list(R.ravel()))
    print(trans_matrix.GetMatrix())

    trans_inverse.DeepCopy(trans_matrix)
    trans_inverse.Inverse()

    count = 0
    # slice along normal
    a1 = r_max#*b1
    a2 = r_max#*b2
    start_pt = np.copy(neck_start_pt)
    result = itertools.cycle([(1.,1.), (-1., 1.), (-1., -1.), (1., -1.)])
    end_pt = 0.0
    #slice until there some reasonable overlap at the end
    while(np.linalg.norm(neck_projection+(count-0.5)*dx*n) < v_mag):
        print(np.linalg.norm(neck_projection+(count-0.5)*dx*n), v_mag)
        step_dx = count*dx*n
        for i in range(4):
            box_dir = next(result)
            # point to define bounds
            #end_pt = start_pt + box_dir[0]*a1 + box_dir[1]*a2 + step_dx 
            dims = np.array([[box_dir[0]*r_max, box_dir[1]*r_max, (count+1)*dx],
                            [0.0, 0.0, count*dx]])
            dims_min = list(dims.min(axis=0))
            dims_max = list(dims.max(axis=0))
            print(dims_min, dims_max)
            
            #planes = vtk.vtkBox()
            #planes.SetBounds (dims_min[0], dims_max[0], dims_min[1], dims_max[1], dims_min[2], dims_max[2])
            #planes.SetTransform(trans_inverse)
            surface = vtk.vtkPolyData()
            surface.DeepCopy(input_surface)
            
            #surface = Input
            for j in range(3):
                for k in range(2):
                    plane = vtk.vtkPlane()
                    normal = [0.0,0.0,0.0]
                    if (k == 0):
                        normal[j] = -1.0
                        plane.SetOrigin(dims_min)
                    else:
                        normal[j] = 1.0
                        plane.SetOrigin(dims_max);
                    plane.SetNormal(normal)
                    plane.SetTransform(trans_inverse)
                    #plane.SetTransform(trans_matrix)
                    clipper = vtk.vtkTableBasedClipDataSet()
                    clipper.SetInputData(surface)
                    clipper.SetClipFunction(plane)
                    #clipper.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
                    clipper.InsideOutOn()
                    #clipper.SetMergeTolerance(1.0E-6)
                    clipper.Update()
                    #print(clipper.GetMergeTolerance())
                    surface = clipper.GetOutput()

            geom = vtk.vtkGeometryFilter()
            geom.SetInputData(surface)
            geom.Update()
            

            writer = vmtkscripts.vmtkSurfaceWriter()
            writer.OutputFileName = os.path.join(args.out_dir, "{0}_{1}_quad_{2}.vtp".format(args.out_file, count,i))
            writer.Input = geom.GetOutput()
            writer.Execute()
            
            #test = vtk.vtkCubeSource()
            #test.SetBounds (dims_min[0], dims_max[0], dims_min[1], dims_max[1], dims_min[2], dims_max[2])
            
            #trans_cube = vtk.vtkTransformPolyDataFilter()
            #trans_cube.SetInputConnection(test.GetOutputPort())
            #trans_cube.SetTransform(trans_matrix)
            #trans_cube.Update()
            
            #writer2 = vmtkscripts.vmtkSurfaceWriter()
            #writer2.OutputFileName = os.path.join(args.out_dir, "{0}_{1}_quad{2}_box.vtp".format(args.out_file, count,i))
            #writer2.Input = trans_cube.GetOutput()
            #writer2.Execute()
            
        count+=1


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='estimate vertices  for uniform point distribution')
    parser.add_argument("-i", dest="surface_file", required=True, help="input probed surface file", metavar="FILE")
    parser.add_argument("-c", dest="centerlines", required=True, help="dome centerlines", metavar="FILE")
    parser.add_argument("-d", dest="out_dir", required=True, help="output dir with clipped surfaces", metavar="FILE")
    parser.add_argument("-o", dest="out_file", required=True,
                        help="output filename for clipped surfaces", metavar="FILE",
                        default = "case_dome_normals_probe_clip")
    parser.add_argument("-t", '--thickness', dest="slice_thickness",  type=float, help='slice thickness of surface ', default=0.5625)
    parser.add_argument("--centroid", dest="centroid", required=True, help="string of comma space seperated numbers to define neck centroid", metavar="str")
    args = parser.parse_args()
    #print(args)
    Execute(args)




