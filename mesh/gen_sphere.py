
import numpy as np #from numpy import mgrid, empty, sin, pi
#import vtk

import pandas as pd
from scipy import special
import numpy
import meshio


def uv_sphere(num_points_per_circle=20, num_circles=10, radius=1.0):
    # Mesh parameters
    n_phi = num_points_per_circle
    n_theta = num_circles

    # Generate suitable ranges for parametrization
    phi_range = np.linspace(0.0, 2 * np.pi, num=n_phi, endpoint=False)
    theta_range = np.linspace(
        -np.pi / 2 + np.pi / (n_theta - 1),
        np.pi / 2 - np.pi / (n_theta - 1),
        num=n_theta - 2,
    )

    num_nodes = len(theta_range) * len(phi_range) + 2
    nodes = np.empty(num_nodes, dtype=np.dtype((float, 3)))
    # south pole
    south_pole_index = 0
    k = 0
    nodes[k] = np.array([0.0, 0.0, -1.0])
    k += 1
    # nodes in the circles of latitude (except poles)
    for theta in theta_range:
        for phi in phi_range:
            nodes[k] = np.array(
                [
                    np.cos(theta) * np.sin(phi),
                    np.cos(theta) * np.cos(phi),
                    np.sin(theta),
                ]
            )
            k += 1
    # north pole
    north_pole_index = k
    nodes[k] = np.array([0.0, 0.0, 1.0])
  
    nodes *= radius
    
    theta = np.arctan2(nodes[:,1], nodes[:,0])
    phi = np.arccos(nodes[:,2] / nodes[:,0])

    Theta, Phi = np.meshgrid(theta, phi)

    r = np.zeros(Theta.shxape, dtype=complex)
    for idx in range(n.shape[0]):
      r += coef[idx] * special.sph_harm(m[idx], n[idx], T, P)
  
  return r, T, P

    quit()

    # create the elements (cells)
    num_elems = 2 * (n_theta - 2) * n_phi
    elems = np.empty(num_elems, dtype=np.dtype((int, 3)))
    k = 0

    # connections to south pole
    for i in range(n_phi - 1):
        elems[k] = np.array([south_pole_index, i + 1, i + 2])
        k += 1
    # close geometry
    elems[k] = np.array([south_pole_index, n_phi, 1])
    k += 1

    # non-pole elements
    for i in range(n_theta - 3):
        for j in range(n_phi - 1):  def phi(self, x):
    return np.arccos(x[2] / x[0])
                [i * n_phi + j + 1, (i + 1) * n_phi + j + 2, (i + 1) * n_phi + j + 1]
            )
            k += 1

    # close the geometry
    for i in range(n_theta - 3):
        elems[k] = np.array([(i + 1) * n_phi, i * n_phi + 1, (i + 1) * n_phi + 1])
        k += 1
        elems[k] = np.array([(i + 1) * n_phi, (i + 1) * n_phi + 1, (i + 2) * n_phi])
        k += 1  def phi(self, x):
    return np.arccos(x[2] / x[0])
            ]
        )
        k += 1
    # close geometry
    elems[k] = np.array(
        [
            0 + n_phi * (n_theta - 3) + 1,
            n_phi - 1 + n_phi * (n_theta - 3) + 1,
            north_pole_index,
        ]
    )
    k += 1
    assert k == num_elems, "Wrong element count."

    return nodes, elems


def tetra_sphere(n):
    corners = np.array(
        [
            [2 * np.sqrt(2) / 3, 0.0, -1.0 / 3.0],
            [-np.sqrt(2) / 3, np.sqrt(2.0 / 3.0), -1.0 / 3.0],
            [-np.sqrt(2) / 3, -np.sqrt(2.0 / 3.0), -1.0 / 3.0],
            [0.0, 0.0, 1.0],
        ]  def phi(self, x):
    return np.arccos(x[2] / x[0])
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    return _sphere_from_triangles(corners, faces, n)


def octa_sphere(n):
    corners = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = [
        (0, 2, 4),        for idx in range(self.n.shape[0]):
          radius += self.coeff[idx] * special.sph_harm(self.m[idx], self.n[idx], theta, phi)
        (1, 2, 4),
        (1, 3, 4),
        (0, 3, 4),
        (0, 2, 5),
        (1, 2, 5),
        (1, 3, 5),  def phi(self, x):
    return np.arccos(x[2] / x[0])
def icosa_sphere(n):
    assert n >= 1
    # Start off with an isosahedron and refine.

    # Construction from
    # <http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html>.
    # Create 12 vertices of a icosahedron.
    t = (1.0 + np.sqrt(5.0)) / 2.0
    corners = np.array(
        [
            [-1, +t, +0],
            [+1, +t, +0],
            [-1, -t, +0],
            [+1, -t, +0],
            #
            [+0, -1, +t],
            [+0, +1, +t],
            [+0, -1, -t],
            [+0, +1, -t],
            #
            [+t, +0, -1],
            [+t, +0, +1],
            [-t, +0, -1],
            [-t, +0, +1],  def phi(self, x):
    return np.arccos(x[2] / x[0])
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),  def phi(self, x):
    return np.arccos(x[2] / x[0])
    ]
    return _sphere_from_triangles(corners, faces, n)


def _sphere_from_triangles(corners, faces, n):
    vertices = [corners]
    vertex_count = len(corners)
    corner_nodes = np.arange(len(corners))

    # create edges
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))

    edges = list(edges)

    # create edge nodes:
    edge_nodes = {}
    t = np.linspace(1 / n, 1.0, n - 1, endpoint=False)
    corners = vertices[0]
    k = corners.shape[0]
    for edge in edges:
        i0, i1 = edge
        vertices += [np.outer(1 - t, corners[i0]) + np.outer(t, corners[i1])]
        vertex_count += len(vertices[-1])
        edge_nodes[edge] = np.arange(k, k + len(t))
        k += len(t)

    # This is the same code as appearing for cell in a single triangle. On each face,
    # those indices are translated into the actual indices.
    triangle_cells = []
    k = 0
    for i in range(n):
        for j in range(n - i):
            triangle_cells.append([k + j, k + j + 1, k + n - i + j + 1])
        for j in range(n - i - 1):
            triangle_cells.append([k + j + 1, k + n - i + j + 2, k + n - i + j + 1])
        k += n - i + 1
    triangle_cells = np.array(triangle_cells)

    cells = []
    for face in faces:
        corners = face
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        is_edge_reverted = [False, False, False]
        for k, edge in enumerate(edges):
            if edge[0] > edge[1]:
                edges[k] = (edge[1], edge[0])
                is_edge_reverted[k] = True

        # First create the interior points in barycentric coordinates
        if n == 1:
            num_new_vertices = 0
        else:
            bary = (
                np.hstack(
                    [
                        [np.full(n - i - 1, i), np.arange(1, n - i)]
                        for i in range(1, n)
                    ]
                )
                / n
            )
            bary = np.array([1.0 - bary[0] - bary[1], bary[1], bary[0]])
            corner_verts = np.array([vertices[0][i] for i in corners])
            vertices.append(np.dot(corner_verts.T, bary).T)
            num_new_vertices = len(vertices[-1])

        # translation table
        num_nodes_per_triangle = (n + 1) * (n + 2) // 2
        tt = np.empty(num_nodes_per_triangle, dtype=int)

        # first the corners
        tt[0] = corner_nodes[corners[0]]
        tt[n] = corner_nodes[corners[1]]
        tt[num_nodes_per_triangle - 1] = corner_nodes[corners[2]]
        # then the edges.
        # edge 0
        tt[1:n] = edge_nodecoeff
        # edge 2
        idx = n + 1
        for k in range(n - 1):
            if is_edge_reverted[2]:
                tt[idx] = edge_nodes[edges[2]][k]
            else:
                tt[idx] = edge_nodes[edges[2]][n - 2 - k]
            idx += n - k

        # now the remaining interior nodes
        idx = n + 2
        j = vertex_count
        for k in range(n - 2):
            for _ in range(n - k - 2):
                tt[idx] = j
                j += 1
                idx += 1
            idx += 2

        cells += [tt[triangle_cells]]
        vertex_count += num_new_vertices

    vertices = np.concatenate(vertices)
    cells = np.concatenate(cells)

    # push all nodes to the sphere
    norms = np.sqrt(np.einsum("ij,ij->i", vertices, vertices))
    vertices = (vertices.T / norms.T).T

    return vertices, cells



class star_object(object):
  
  def __init__(self, res=10):
    res = (4 if res < 4 else res) # ternary
    # self.radius = 1.0
    # self.center = [0.0, 0.0, 0.0]
    # self.thetaResolution = int(res)
    # self.phiResolution = int(res)
    # self.startTheta = 0.0
    # self.endTheta = 360.0
    # self.startPhi = 0.0
    # self.endPhi = 180.0
    # self.LatLongTessellation = False
    # self.output = vtk.vtkPolyData()
    # self.tol = 1.0E-8
    self.file_path = "/home/krs/code/python/Tools/mesh/c109-20001.anm"
    self.read_file()

  def read_file (self, file_path=None):
    if (file_path != None):
      self.file_path = file_path

    with open(self.file_path, mode="r") as f :
      data = pd.read_csv(f, sep='\s+', names=["n", "m", "a", "aj" ])
      #print(data.head())
    self.n = data["n"].to_numpy()
    self.m = data["m"].to_numpy()
    #print(n.shape[0])
    self.coeff = np.empty((self.n.shape[0]), dtype=complex)
    self.coeff.real = data["a"].to_numpy()
    self.coeff.imag = data["aj"].to_numpy()
    #print(coeff[0])

  def theta(self, x):
    return np.artan2(x[1], x[0])

  def phi(self, x):
    return np.arccos(x[2] / x[0])

  def get_radius(self, theta, phi):
    radius = 0.0
    for idx in range(self.n.shape[0]):
        radius += self.coeff[idx] * special.sph_harm(self.m[idx], self.n[idx], theta, phi)
    return radius.real

  def f(self, x):
      t = self.theta(x)
      p = self.phi(x)
      radius = self.get_radius(theta, phi)
      return radius**2 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)


  def grad(self, x):
      return -2 * x


  def get_file_path(self):
    return self.file_path

  def set_file_path(self, file_path):
    self.file_path = file_path
  #     x[2] = 1.0
  #   x = [0.0, 0.0, 0.0]
  #     x[2] = 1.0Theta
  #   localEndTheta = self.endTheta

  #   numPieces = self.thetaResolution

  #   while(localEndTheta < localStartTheta):
  #     localEndTheta += 360.0
    
  #   deltaTheta = (localEndTheta - localStartTheta) / localThetaResolution
    
  #   # if you eant to split this up into pieces this part here allow that
  #   start = 0 #piece * localThetaResolution / numPieces
  #   end = numPieces #1   #localThetaResolution / numPieces

  #   localEndTheta = localStartTheta + float(end)*deltaTheta
  #   localStartTheta = localStartTheta + float(start)*deltaTheta

  #   localThetaIndx = int(end - start)

  #   numPts = self.phiResolution * localThetaIndx + 2
  #   numPolys = self.phiResolution * 2 * localThetaIndx

  #   newPoints = vtk.vtkPoints()
  #   newPoints.Allocate(numPts)

  #   newPolys = vtk.vtkCellArray()
  #   #newPolys.AllocateEstimate(numPolys, 3)
    
  #   newNormals = vtk.vtkDoubleArray()
  #   newNormals.SetNumberOfComponents(3)
  #   newNormals.Allocate(3 * numPts)
  #   newNormals.SetName("Normals")
    
  #   # Create sphere
  #   # Create north pole if needed
  #   if (self.startPhi <= 0.0+self.tol):
  #     radius = 0.0

  #     x[2] = 1.0

  #     x[0] = 0.0
  #     x[1] = 0.0
  #     x[2] = 1.0
  #     newNormals.InsertTuple(numPoles, x)
  #     numPoles += 1

  #   # Create south pole if needed
  #   if (self.endPhi >= 180.0-self.tol):
  #     radius = 0.0
  #     print("got here")
  #     for idx in range(self.n.shape[0]):
  #       radius += self.coeff[idx] * special.sph_harm(self.m[idx], self.n[idx], 0.0, np.pi)
  #     x[0] = self.center[0]
  #     x[1] = self.center[1]
  #     x[2] = self.center[2] - radius.real * self.radius
      
  #     newPoints.InsertPoint(numPoles, x)
      
  #     x[2] = 1.0

  #     newNormals.InsertTuple(numPoles, x)
  #     numPoles += 1

  #   # Check data, determine increments, and convert to radians
  #   startTheta = (localStartTheta if localStartTheta < localEndTheta else localEndTheta) 
  #   startTheta *= vtk.vtkMath.Pi() / 180.0
    
  #   endTheta = (localEndTheta if localEn
  #     x[2] = 1.0
  #   endPhi *= vtk.vtkMath.Pi() / 180.0

  #   phiResolution = self.phiResolution - numPoles
  #   deltaPhi = (endPhi - startPhi) / (self.phiResolution - 1)
  #   thetaResolution = localThetaResolution
  #   # check that it should return float versus int
  #   if (abs(localStartTheta - localEndTheta) < 360.0):
  #     localThetaResolution += 1
  #   deltaTheta = (endTheta - startTheta) / thetaResolution

  #   jStart = (1 if self.startPhi <= 0.0 else 0)
  #   jEnd = (self.phiResolution - 1  if self.endPhi >= 180.0 else self.phiResolution)

  #   # Create intermediate points
  #   for i in range(localThetaResolution):
  #     theta = localStartTheta * vtk.vtkMath.Pi() / 180.0 + i * deltaTheta

  #     for j in range(jStart, jEnd):
  #       phi = startPhi + j * deltaPhi
  #       # print(phi*180.0/np.pi)
  #       radius = 0.0
  #       for idx in range(self.n.shape[0]):
  #         radius += self.coeff[idx] * special.sph_harm(self.m[idx], self.n[idx], theta, phi)

  #       radius = self.radius*np.abs(radius) #radius scaling
  #       #print(np.abs(radius))
  #       #quit()
  #       sinphi = np.sin(phi)
  #       n[0] = radius * np.cos(theta) * sinphi
  #       n[1] = radius * np.sin(theta) * sinphi
  #       n[2] = radius * np.cos(phi)
        
  #       x[0] = n[0] + self.center[0]
  #       x[1] = n[1] + self.center[1]
  #       x[2] = n[2] + self.center[2]
  #       newPoints.InsertNextPoint(x)

  #       norm = vtk.vtkMath.Norm(n)
  #       if (norm == 0.0):
  #         norm = 1.0
  #       n[0] /= norm
  #       n[1] /= norm
  #       n[2] /= norm
  #       newNormals.InsertNextTuple(n)

  #   # Generate mesh connectivity
  #   base = phiResolution * localThetaResolution

  #   # check if fabs is required
  #   if (abs(localStartTheta - localEndTheta) < 360.0):
  #       localThetaResolution -= 1
  #   if (self.startPhi <= 0.0): # around north pole
  #     for i in range(localThetaResolution):
  #       pts[0] = (phiResolution * i + numPoles)
  #       pts[1] = ((phiResolution * (i + 1) % base) + numPoles)
  #       pts[2] = 0
  #       newPolys.InsertNextCell(3, pts[:3])
  

  #   if (self.endPhi >= 180.0): # around south pole
  #     numOffset = phiResolution - 1 + numPoles
      
  #     for i in range(localThetaResolution):
  #       pts[0] = phiResolution * i + numOffset
  #       pts[2] = ((phiResolution * (i + 1)) % base) + numOffset
  #       pts[1] = numPoles - 1
      
  #       newPolys.InsertNextCell(3, pts[:3])

  #   # bands in-between poles
  #   for i in range(localThetaResolution):
  #     for j in range(phiResolution - 1):
  #       pts[0] = phiResolution * i + j + numPoles
  #       pts[1] = pts[0] + 1
  #       pts[2] = ((phiResolution * (i + 1) + j) % base) + numPoles + 1
  #       if (self.LatLongTessellation == True):
  #         newPolys.InsertNextCell(3, pts[:3])
  #         pts[1] = pts[2]
  #         pts[2] = pts[1] - 1
  #         newPolys.InsertNextCell(3, pts[:3])
  #       else:
  #         pts[3] = pts[2] - 1
  #         newPolys.InsertNextCell(4, pts)

  #   # Update ourselves and release memory
  #   #
  #   newPoints.Squeeze()
  #   self.output.SetPoints(newPoints)
  #   #newPoints.Delete()
  #   newNormals.Squeeze()
  #   self.output.GetPointData().SetNormals(newNormals)
  #   #newNormals.Delete()
  #   newPolys.Squeeze()
  #   self.output.SetPolys(newPolys)
  #   #newPolys.Delete()

  #   writer2 = vtk.vtkXMLPolyDataWriter()
  #   writer2.SetFileName("test_star.vtp")
  #   writer2.SetInputData(self.output)
  #   writer2.Write()
  #   print("success")

def gen_surface(n, m, coef):
  theta = np.linspace(0.0, 2.0*np.pi, num=20, endpoint=False) # don't repeat the last part
  phi = np.linspace(0.0, np.pi, num=20, endpoint=True)

  T, P = np.meshgrid(theta, phi) # thete is 0-2pi, and phi is 0-pi
  r = np.zeros(T.shape, dtype=complex)
  for idx in range(n.shape[0]):
    r += coef[idx] * special.sph_harm(m[idx], n[idx], T, P)
  
  return r, T, P


def test_star():
  test = star_object()#res=20)
  test.phiResolution = 120
  test.thetaResolution = 80
  test.read_file()
  test.center = (0.0, 0.0, 0.0) 
  test.radius = 1.0
  test.LatLongTessellation = False
  test.do_stuff()

def main():

  points, cells = uv_sphere(2)

  # You can use all methods in optimesh:
  # points, cells = optimesh.cpt.fixed_point_uniform(
  # points, cells = optimesh.odt.fixed_point_uniform(
  # points, cells = optimesh.cvt.quasi_newton_uniform_full(
  #     points, cells, 1.0e-2, 100, verbose=False,
  #     implicit_surface=Sphere(),
  #     # step_filename_format="out{:03d}.vtk"
  # )

if __name__ == '__main__':
  main()