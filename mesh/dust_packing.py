# create box with a bunch of sphere shaped particles in it.

import os
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import miniball
import numpy as np
import pickle

# import pooch
# from gen_sphere import particle_generator


input_dir = "/media/store/krs/particles/C109-sand"
all_files = glob.glob(os.path.join(input_dir, "*.vtp"))

reader = vtk.vtkXMLPolyDataReader()
com = vtk.vtkCenterOfMass()
com.SetUseScalarsAsWeights(False)

data = {}

for file_path in all_files:
    reader.SetFileName(file_path)
    reader.Update()
    com.SetInputConnection(reader.GetOutputPort())
    com.Update()
    center = np.array(com.GetCenter())

    pts = reader.GetOutput().GetPoints()

    # gives nx3
    pts_np = vtk.util.numpy_support.vtk_to_numpy(pts.GetData())

    # If I take the distance from the centroid and select the top 2000
    # samples then I get pretty darn close the solution of doing the miniball
    # of the whole domain.
    # so as long as this part is cheap, then the minball doesn't have to be super fast
    dist = np.sqrt(np.sum(np.power(pts_np - center, 2), axis=1))
    ind = np.argpartition(dist, -2000)[-2000:]
    c, r2 = miniball.get_bounding_ball(pts_np[ind, :])  # [0:5000, :])
    r = np.sqrt(r2)
    vol = 4.0 / 3.0 * np.pi * r ** 3
    # print(c.shape, r.shape, vol.shape)
    data[file_path] = dict(centroid=list(c), radius=r, volume=vol)


pickle.dump(data, open("C109-sand_spheres.pkl", "wb"))

print("all done")
