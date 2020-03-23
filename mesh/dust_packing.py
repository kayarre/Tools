# create box with a bunch of sphere shaped particles in it.

import os
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import miniball

# import pooch
# from gen_sphere import particle_generator


input_dir = "/media/store/krs/particles/C109-sand"
all_files = glob.glob(os.path.join(input_dir, "*.vtp"))

reader = vtk.vtkXMLPolyDataReader()
com = vtk.vtkCenterOfMass()

com.SetUseScalarsAsWeights(False)
com.Update()
for test in all_files:
    # print(test)
    # quit()
    reader.SetFileName(test)
    reader.Update()
    com.SetInputConnection(reader.GetOutputPort())
    com.Update()

    pts = reader.GetOutput().GetPoints()
    print(com.GetCenter())
    quit()
    # gives nx3
    test = vtk.util.numpy_support.vtk_to_numpy(pts.GetData())
    c, r2 = miniball.get_bounding_ball(test)  # [0:5000, :])

    print(c, r2)
    quit()
    print("all done")
