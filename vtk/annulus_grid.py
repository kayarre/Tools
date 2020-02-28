# Authors: Eric Jones <eric at enthought dot com>
#          Prabhu Ramachandran <prabhu at aero dot iitb dot ac dot in>
# Copyright (c) 2007, Enthought, Inc.
# License: BSD style.

import numpy as np
import vtk

def generate_annulus(r=None, theta=None, z=None):
  """ Generate points for structured grid for a cylindrical annular
      volume.  This method is useful for generating a structured
      cylindrical mesh for VTK (and perhaps other tools).

      Parameters
      ----------
      r : array : The radial values of the grid points.
                  It defaults to linspace(1.0, 2.0, 11).

      theta : array : The angular values of the x axis for the grid
                      points. It defaults to linspace(0,2*pi,11).

      z: array : The values along the z axis of the grid points.
                  It defaults to linspace(0,0,1.0, 11).

      Return
      ------
      points : array
          Nx3 array of points that make up the volume of the annulus.
          They are organized in planes starting with the first value
          of z and with the inside "ring" of the plane as the first
          set of points.  The default point array will be 1331x3.
  """
  # Default values for the annular grid.
  if r is None:
    r = np.linspace(1.0, 2.0, 11)
  if theta is None:
    theta = np.linspace(0, 2*np.pi, 11)
  if z is None:
    z = np.linspace(0.0, 1.0, 11)

  # Find the x values and y values for each plane.
  x_plane = (np.cos(theta)*r[:,None]).ravel()
  y_plane = (np.sin(theta)*r[:,None]).ravel()

  # Allocate an array for all the points.  We'll have len(x_plane)
  # points on each plane, and we have a plane for each z value, so
  # we need len(x_plane)*len(z) points.
  points = np.empty([len(x_plane)*len(z),3])

  # Loop through the points for each plane and fill them with the
  # correct x,y,z values.
  start = 0
  for z_plane in z:
    end = start + len(x_plane)
    # slice out a plane of the output points and fill it
    # with the x,y, and z values for this plane.  The x,y
    # values are the same for every plane.  The z value
    # is set to the current z
    plane_points = points[start:end]
    plane_points[:,0] = x_plane
    plane_points[:,1] = y_plane
    plane_points[:,2] = z_plane
    start = end

  return points

def main():
  # Make the data.
  dims = (51, 25, 25)
  # Note here that the 'x' axis corresponds to 'theta'
  theta = np.linspace(0, 2*np.pi, dims[0])
  # 'y' corresponds to varying 'r'
  r = np.linspace(1, 10, dims[1])
  z = np.linspace(0, 5, dims[2])
  pts = generate_annulus(r, theta, z)
  vtk_pts = vtk.vtkPoints()
  for pt in pts:
    #print(pt)
    vtk_pts.InsertNextPoint(pt)
  # Uncomment the following if you want to add some noise to the data.
  #pts += np.random.randn(dims[0]*dims[1]*dims[2], 3)*0.04
  sgrid = vtk.vtkStructuredGrid()
  sgrid.SetDimensions(dims)

  sgrid.SetPoints(vtk_pts)
  scalar_arr = vtk.vtkDoubleArray()
  scalar_arr.SetNumberOfComponents(1)
  scalar_arr.SetName("distance")
  s = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
  for s_ in s:
    scalar_arr.InsertNextTuple([s_])
  #print(s.shape)
  sgrid.GetPointData().AddArray(scalar_arr)
  # sgrid.point_data.scalars.name = 'scalars'

  # Uncomment the next two lines to save the dataset to a VTK XML file.
  writer = vtk.vtkXMLStructuredGridWriter()
  writer.SetFileName("test.vts")
  writer.SetInputData(sgrid)
  writer.Write()
  #w = tvtk.XMLStructuredGridWriter(input=sgrid, file_name='sgrid.vts')
  #w.write()


if __name__ == '__main__':
  main()