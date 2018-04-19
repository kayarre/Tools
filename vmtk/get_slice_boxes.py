

import vtk
import numpy as np
import itertools
import os


# clip boxes along the surface

# this file is the surface to clip
file_name =  "/home/sansomk/caseFiles/mri/VWI_proj/case1/VWI_analysis/case1_dome_normals_probe.vtp"

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(file_name)
reader.Update()

# this is the centerline file
file_name2 = "/home/sansomk/caseFiles/mri/VWI_proj/case1/vmtk_dsa/case1_dome_ctr_smooth_resample_geom.vtp"
reader2 = vtk.vtkXMLPolyDataReader()
reader2.SetFileName(file_name2)
reader2.Update()



file_name2_out = "/home/sansomk/caseFiles/mri/VWI_proj/case1/VWI_analysis/case1_dome_normals_probe_clip_info.vtp"
file_path = "/home/sansomk/caseFiles/mri/VWI_proj/case1/VWI_analysis"

#vec = np.array(reader.GetOutput().GetPointData().GetArray("NRRDImage").GetTuple(i))


dx = 0.5625
neck_centroid = np.array([10.92781557999258, 20.516195612629566, 27.53231491629533])


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


n_pts = reader2.GetOutput().GetNumberOfPoints()
pt1 = np.array(reader2.GetOutput().GetPoint(0)) # start point
pt2 = np.array(reader2.GetOutput().GetPoint(n_pts-1)) # end point

#print(pt1, pt2)
v =  pt2 - pt1 #pt1 - pt2
v_mag = np.linalg.norm(v)
n = v / v_mag
print("should be 1.0", np.linalg.norm(n), n)

b1, b2 = hughes_moeller(n) #orthogonal basis


#Get  maximum radius
radius_range = [0.0, 0.0]
reader2.GetOutput().GetPointData().GetArray("MaximumInscribedSphereRadius").GetRange(radius_range, 0)
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
        #print(dims_min, dims_max)
        
        #planes = vtk.vtkBox()
        #planes.SetBounds (dims_min[0], dims_max[0], dims_min[1], dims_max[1], dims_min[2], dims_max[2])
        #planes.SetTransform(trans_inverse)
        surface = vtk.vtkPolyData()
        surface.DeepCopy(reader.GetOutput())
        #surface = reader.GetOutput()
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
        
        #clipper = vtk.vtkCutter()
        #clipper.SetInputConnection(reader.GetOutputPort())
        #clipper.SetCutFunction(planes)
        ##clipper.SetValue(0.0)
        ##clipper.InsideOutOn()
        #clipper.Update()
        
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputConnection(geom.GetOutputPort())
        writer.SetFileName(os.path.join(file_path, "case1_dome_normals_probe_clip_slice{0}_quad{1}.vtp".format(count,i)))
        writer.Update()
        
        #test = vtk.vtkCubeSource()
        #test.SetBounds (dims_min[0], dims_max[0], dims_min[1], dims_max[1], dims_min[2], dims_max[2])
        
        #trans_cube = vtk.vtkTransformPolyDataFilter()
        #trans_cube.SetInputConnection(test.GetOutputPort())
        #trans_cube.SetTransform(trans_matrix)
        
        #writer2 = vtk.vtkXMLPolyDataWriter()
        #writer2.SetInputConnection(trans_cube.GetOutputPort())
        #writer2.SetFileName(os.path.join(file_path,
        #                                "case1_dome_normals_probe_clip_slice{0}_quad{1}_box.vtp".format(count,i)))
        #writer2.Update()
        
    #break
    count+=1
                           











#lines = np.empty((n_cells, n_pts))
#pts = np.empty((n_cells, 3))
#da = reader.GetOutput().GetPointData().GetArray("NRRDImage")

#for i in range(n_cells):
    #cellids = reader.GetOutput().GetCell(i).GetPointIds()
    ##n_pts = cell.GetNumberOfPoints()
    #for j in range(n_pts):
        #if(j == n_pts // 2):
            #pts[i,:] = np.array(reader.GetOutput().GetPoint(cellids.GetId(j)))
        
        #lines[i, j] = reader.GetOutput().GetPointData().GetArray("NRRDImage").GetTuple(cellids.GetId(j))[0]
        

#ln_avg = np.average(lines, axis=1)

#ln_avg_norm = ln_avg / ln_avg.max()




##Create the tree
#pointLocator = vtk.vtkPointLocator()
#pointLocator.SetDataSet(reader2.GetOutput())
#pointLocator.BuildLocator()

#array = vtk.vtkDoubleArray()
#array.SetNumberOfComponents(n_pts)
#array.SetName("raw")
#array.SetNumberOfTuples(n_cells)

#avg = vtk.vtkDoubleArray()
#avg.SetNumberOfComponents(1)
#avg.SetName("avg")
#avg.SetNumberOfTuples(n_cells)

#avg_norm = vtk.vtkDoubleArray()
#avg_norm.SetNumberOfComponents(1)
#avg_norm.SetName("normalized")
#avg_norm.SetNumberOfTuples(n_cells)

#for i in range(n_cells):
    #surf_id = pointLocator.FindClosestPoint(pts[i])
    ##print(ln_avg.shape)
    #avg.SetValue(surf_id, ln_avg[i])
    #array.SetTuple(surf_id, list(lines[i,:]))
    #avg_norm.SetValue(surf_id, ln_avg_norm[i])


#reader2.GetOutput().GetPointData().AddArray(avg)

#reader2.GetOutput().GetPointData().AddArray(array)
#reader2.GetOutput().GetPointData().AddArray(avg_norm)


#writer = vtk.vtkXMLPolyDataWriter()
#writer.SetFileName(file_name2_out)
#writer.SetInputConnection(reader2.GetOutputPort())
#writer.Update()




#for i in range(polydata.GetNumberOfPoints()):
    #array.InsertNextValue(somecomputedvalue)

#polydata.GetPointData().AddArray(array);


##Find the closest points to TestPoint
#closestPoint = (0.0, 0.0, 0.0,) # the coordinates of the closest point will be returned here
#closestPointDist2 = 0.0 #the squared distance to the closest point will be returned here
#cellId  = vtk.vtkIdType() # the cell id of the cell containing the closest point will be returned here
#subId = 0 #this is rarely used (in triangle strips only, I believe)
#pointLocator.FindClosestPoint(testPoint, closestPoint, cellId, subId, closestPointDist2);

#print( "Coordinates of closest point: {0}".format(closestPoint))
#print("Squared distance to closest point: {0}".format(closestPointDist2))
#print("CellId: {0}".format(cellId))
