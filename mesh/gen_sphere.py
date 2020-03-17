import numpy as np  # from numpy import mgrid, empty, sin, pi
import pandas as pd
from scipy import special
import meshio
import pooch
import os
import optimesh
import sys

np.set_printoptions(threshold=sys.maxsize)


def Cart_to_Spherical_np(xyz):
    # physics notation
    # print(xyz.shape)
    # quit()
    ptsnew = np.zeros(xyz.shape)  # np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[0] ** 2 + xyz[1] ** 2  # r
    ptsnew[0] = np.sqrt(xy + xyz[2] ** 2)
    ptsnew[1] = np.arctan2(  # theta
        np.sqrt(xy), xyz[2]
    )  # for elevation angle defined from Z-axis down
    # ptsnew[1] = np.arctan2(  # theta
    #     xyz[2], np.sqrt(xy)
    # )  # for elevation angle defined from XY-plane up
    ptsnew[2] = np.arctan2(xyz[1], xyz[0])  # phi
    return ptsnew


def appendSpherical_np(xyz):
    # physics notation
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)  # r
    ptsnew[:, 4] = np.arctan2(  # theta
        np.sqrt(xy), xyz[:, 2]
    )  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])  # phi
    return ptsnew


def uv_sphere(
    num_points_per_circle=20, num_circles=10, radius=1.0,
):
    # Mesh parameters
    n_phi = num_points_per_circle
    n_theta = num_circles

    # physics notation
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
        for j in range(n_phi - 1):
            elems[k] = np.array(
                [i * n_phi + j + 1, i * n_phi + j + 2, (i + 1) * n_phi + j + 2]
            )
            k += 1
            elems[k] = np.array(
                [i * n_phi + j + 1, (i + 1) * n_phi + j + 2, (i + 1) * n_phi + j + 1]
            )
            k += 1

    # close the geometry
    for i in range(n_theta - 3):
        elems[k] = np.array([(i + 1) * n_phi, i * n_phi + 1, (i + 1) * n_phi + 1])
        k += 1
        elems[k] = np.array([(i + 1) * n_phi, (i + 1) * n_phi + 1, (i + 2) * n_phi])
        k += 1

    # connections to the north pole
    for i in range(n_phi - 1):
        elems[k] = np.array(
            [
                i + 1 + n_phi * (n_theta - 3) + 1,
                i + n_phi * (n_theta - 3) + 1,
                north_pole_index,
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
        ]
    )
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
        (0, 2, 4),
        (1, 2, 4),
        (1, 3, 4),
        (0, 3, 4),
        (0, 2, 5),
        (1, 2, 5),
        (1, 3, 5),
    ]
    return _sphere_from_triangles(corners, faces, n)


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
            [-t, +0, +1],
        ]
    )

    faces = [
        (0, 11, 5),
        (0, 5, 1),
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
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
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
                    [[np.full(n - i - 1, i), np.arange(1, n - i)] for i in range(1, n)]
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
        tt[1:n] = edge_nodes[edges[0]]
        if is_edge_reverted[0]:
            tt[1:n] = tt[1:n][::-1]
        #
        # edge 1
        idx = 2 * n
        for k in range(n - 1):
            if is_edge_reverted[1]:
                tt[idx] = edge_nodes[edges[1]][n - 2 - k]
            else:
                tt[idx] = edge_nodes[edges[1]][k]
            idx += n - k - 1
        #
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


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    # assumes nx3 arrays
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def estimate_surface_normals(vertices, faces):

    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    norm_final = normalize_v3(norm)
    return norm_final


class star_object(object):
    def __init__(self, file_path=None):  # , res=10):
        # res = 4 if res < 4 else res  # ternary
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
        if file_path != None:
            self.file_path = file_path
            self.read_file()
        else:
            self.file_path = "/home/krs/code/python/Tools/mesh/c109-20001.anm"
        # self.read_file()

    def read_file(self, file_path=None):
        if file_path != None:
            self.file_path = file_path

        with open(self.file_path, mode="r") as f:
            data = pd.read_csv(f, sep="\s+", names=["n", "m", "a", "aj"])
            # print(data.head())
        self.n = data["n"].to_numpy()
        self.m = data["m"].to_numpy()
        # print(n.shape[0])
        self.coeff = np.empty((self.n.shape[0]), dtype=complex)
        self.coeff.real = data["a"].to_numpy()
        self.coeff.imag = data["aj"].to_numpy()
        # print(coeff[0])
        self.set_norm_coeff()
        # normalizing the coefficients
        # self.set_coeff_to_norm()

    def set_norm_coeff(self):
        self.norm_coeff = self.coeff / self.coeff[0]

    def set_coeff_to_norm(self):
        self.coeff = self.norm_coeff

    def cart_to_spherical_np(self, xyz):
        # physics notation
        ptsnew = np.zeros(xyz.shape)
        xy = xyz[0] ** 2 + xyz[1] ** 2  # r

        ptsnew[0] = np.sqrt(xy + xyz[2] ** 2)
        ptsnew[1] = np.arctan2(  # theta
            np.sqrt(xy), xyz[2]
        )  # for elevation angle defined from Z-axis down
        # ptsnew[:, 1] = np.arctan2(  # theta
        #     xyz[:, 2], np.sqrt(xy)
        # )  # for elevation angle defined from XY-plane up
        # print(ptsnew[1])
        # quit()
        ptsnew[2] = np.arctan2(xyz[1], xyz[0])  # phi

        return ptsnew

    # scalar
    def get_radius(self, theta, phi):
        radius = np.zeros(theta.shape, dtype=np.complex)
        for idx in range(self.n.shape[0]):
            radius += self.coeff[idx] * special.sph_harm(
                self.m[idx], self.n[idx], theta, phi
            )
        return radius.real

    def get_derivatives(self, r, theta, phi):
        radius = np.zeros(theta.shape, dtype=np.complex)
        r_theta = np.zeros(theta.shape, dtype=np.complex)
        r_phi = np.zeros(theta.shape, dtype=np.complex)
        Pnm_phi = np.zeros(theta.shape, dtype=np.complex)
        for idx in range(self.n.shape[0]):

            r_theta += (
                1.0j
                * self.m[idx]
                * self.coeff[idx]
                * special.sph_harm(self.m[idx], self.n[idx], theta, phi)
            )
            # print(theta.shape)
            # print(phi.shape)
            # print(special.sph_harm(self.m[idx], self.n[idx], theta, phi).shape)
            # quit()
            f_nm = np.sqrt(
                ((2.0 * self.n[idx] + 1) * special.factorial(self.n[idx] - self.m[idx]))
                / (4.0 * np.pi * (special.factorial(self.n[idx] + self.m[idx])))
            )

            pre_coef = self.coeff[idx] * f_nm * np.exp(1.0j * self.m[idx] * theta)

            abs_m = abs(self.m[idx])

            if self.n[idx] == 0 and self.m[idx] == 0:
                # r_phi += pre_coef * 0.0
                pass
            elif self.n[idx] == -self.m[idx]:
                # pass
                # print(self.n[idx], self.m[idx])

                # #     # print("got here")
                Pnm_phi = (
                    0.5
                    * np.power(-1, abs_m)
                    / (2.0 * special.factorial(2.0 * self.n[idx] - 1))
                    * special.lpmv(self.n[idx] - 1, self.n[idx], np.cos(phi))
                )
            elif self.n[idx] == self.m[idx]:
                # print(self.n[idx], self.m[idx])
                # print("got here sghadfgsdfg")

                Pnm_phi = self.n[idx] * special.lpmv(
                    self.n[idx] - 1, self.n[idx], np.cos(phi)
                )
            elif self.m[idx] == 0:
                Pnm_phi = -special.lpmv(1, self.n[idx], np.cos(phi))
            # elif self.m[idx] == 1:
            #     Pnm_phi = 0.5 * (
            #         (self.n[idx] + 1)
            #         * self.n[idx]
            #         * special.lpmv(0, self.n[idx], np.cos(phi))
            #     )
            else:
                Pnm_phi = 0.5 * (
                    (self.n[idx] + self.m[idx])
                    * (self.n[idx] - self.m[idx] + 1)
                    * special.lpmv(self.m[idx] - 1, self.n[idx], np.cos(phi))
                    - special.lpmv(self.m[idx] + 1, self.n[idx], np.cos(phi))
                )

            phi_has_nan = np.isnan(pre_coef).any()
            if phi_has_nan:
                raise ValueError(" output has nans")
            array_has_nan = np.isnan(Pnm_phi).any()
            if array_has_nan:
                raise ValueError(" output has nans")
            r_phi += pre_coef * Pnm_phi

            # print(f_nm, pre_coef.min(), pre_coef.max())
            # print(Pnm_phi.min(), Pnm_phi.max())
        # print(r_theta)
        # quit()

        # print(r_theta.min(), r_theta.max())

        #     r.shape, r_theta.shape, r_phi.shape,
        # )
        # print(r.min(), r.max())
        # print(r_phi.min(), r_phi.max())

        S = np.absolute(
            r
            * (
                (r_theta ** 2.0)
                + (r_phi ** 2.0) * (np.sin(phi) ** 2.0)
                + (r ** 2.0) * np.sin(phi)
            )
            ** (0.5)
        )
        t_idx = np.where(S < 0.0000001)
        # print(S[t_idx])
        # print(phi[t_idx])
        # print(r[t_idx])
        # print(r_theta[t_idx])
        # print(r_phi[t_idx])
        # quit()
        array_has_nan = np.isnan(S).any()
        if array_has_nan:
            raise ValueError("output has nans")

        n_x = r * (
            r_theta * np.sin(theta)
            - r_phi * np.sin(phi) * np.cos(phi) * np.cos(theta)
            + r * np.square(np.sin(phi)) * np.cos(theta)
        )  # / S

        n_y = r * (
            -r_theta * np.cos(theta)
            - r_phi * np.sin(phi) * np.cos(phi) * np.sin(theta)
            + r * np.square(np.sin(phi)) * np.sin(theta)
        )  # / S

        # print(n_y)

        n_z = r * (
            r_phi * np.square(np.sin(phi)) + r * np.cos(phi) * np.sin(phi)
        )  # / S
        # print(S.min(), S.max())
        # print(n_x.min(), n_x.max())
        # print(n_y.min(), n_y.max())
        # print(n_z.min(), n_z.max())
        coords = np.stack((n_x.real, n_y.real, n_z.real), axis=0)
        # print(coords.max(axis=1), coords.min(axis=1))
        array_has_nan = np.isnan(coords).any()
        if array_has_nan:
            raise ValueError("output has nans")
        test_norm = np.sqrt(np.sum(np.square(coords), axis=0))
        south_pole = np.where(phi == np.pi)
        north_pole = np.where(phi == 0.0)
        # test_norm[north_pole] =
        # print(coords[:, south_pole[0]].shape)
        coords[:, south_pole[0]] = np.array([[0.0], [0.0], [-1.0]])
        coords[:, north_pole[0]] = np.array([[0.0], [0.0], [1.0]])
        test_norm[south_pole] = 1.0
        test_norm[north_pole] = 1.0

        # quit()
        # zeros = np.where(test_norm < 0.000000001)
        # print(zeros)
        # print(coords.shape)
        # print(coords[:, zeros].shape)

        # # print(test_norm.max(), test_norm.min())
        # # test_norm = np.linalg.norm(coords, axis=0)
        # # print(test_norm.shape, coords.dtype)
        # # print((coords / test_norm).shape)
        # quit()
        return coords  # / test_norm

    # scalar
    def f(self, x):
        sph_coord = self.cart_to_spherical_np(x)
        # note the switch to math notation
        radius = self.get_radius(sph_coord[2], sph_coord[1])
        f_res = radius ** 2 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        return f_res

    def grad(self, x):
        if x.shape[1] != 3:
            if x.shape[1] == 3 and x.shape[0] == 3:
                print("input is ambiguous")
            x = x.T
        else:
            print("there is a weird issue with the grad operator")

        conv_coord = self.cart_to_spherical_np(x.T)
        radius = self.get_radius(conv_coord[2], conv_coord[1])
        # directional dirivatives on the surface
        der_ = self.get_derivatives(radius, conv_coord[2], conv_coord[1])
        # print(der_)
        # implicit derivative
        # print(radius.shape, x.T.shape, der_.shape)
        test = 2.0 * (radius[None, :] * der_ - x.T)
        # print(test)
        # quit()
        return test

    def get_file_path(self):
        return self.file_path

    def set_file_path(self, file_path_str):
        self.file_path = file_path_str


def setup_pooch():
    # Define the Pooch exactly the same (urls is None by default)
    GOODBOY = pooch.create(
        path=pooch.os_cache("particles"),
        base_url="ftp://ftp.nist.gov/pub/bfrl/garbocz/Particle-shape-database/",
        version="0.0.1",
        version_dev="master",
        registry=None,
    )
    # If custom URLs are present in the registry file, they will be set automatically
    GOODBOY.load_registry(os.path.join(os.path.dirname(__file__), "registry.txt"))

    return GOODBOY


def convert_nodes(nodes, star_data):
    sph = Cart_to_Spherical_np(nodes.T)
    r = np.zeros(sph.shape[-1], dtype=np.complex)
    # print(sph[2].shape, sph[1].shape)
    for idx in range(star_data.n.shape[0]):
        # shift phi to be 0 <= phi <= pi
        r += star_data.coeff[idx] * special.sph_harm(
            star_data.m[idx], star_data.n[idx], sph[2], sph[1]  # + np.pi / 2.0
        )
    r_real = r.real
    # print(np.abs(r), r.real)
    # scalar * [nx3] * [n,] = [nx3]
    nodes *= r_real[:, None]

    return nodes


class Sphere:
    def f(self, x):
        # print(x.shape)
        return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    def grad(self, x):
        # print(x.shape)
        return -2 * x


class call_back_test(object):
    def __init__(self, max_num_steps):
        self.avg_quality = np.empty((max_num_steps + 1))

    def test_callback(self, k, mesh):
        self.avg_quality[k] = np.average(mesh.cell_quality)
        print(k)
        return


def main():

    pooch_particles = setup_pooch()
    fname = pooch_particles.fetch("C109-sand/C109-20002.anm")
    print(fname)
    sand = star_object()
    sand.set_file_path(fname)
    sand.read_file()
    # sand.set_coeff_to_norm()

    # print(sand.norm_coeff)

    # points, cells = uv_sphere(20)
    points, cells = icosa_sphere(60)
    conv_pts = convert_nodes(points, sand)
    # print(points, conv_pts.shape)
    # quit()
    estimate_normals = estimate_surface_normals(conv_pts, cells)
    # print(conv_pts.shape, estinmate_normals.shape)
    # quit()
    # print(points, cells)
    tri_cells = [("triangle", cells)]

    # test2 = Sphere()
    # print(points.T.shape)
    # s_pts_2 = test2.f(points.T)
    # s_grad_2 = test2.grad(points.T)
    # print(s_pts_2.shape, s_grad_2.shape)

    # test = star_object(fname)
    # print(points.T.shape)
    # s_pts = test.f(points.T)
    # s_grad = test.grad(points.T)
    # print(s_pts.shape, s_grad.shape)

    mesh = meshio.write_points_cells(
        "test.vtk", conv_pts, tri_cells, point_data={"Normals": estimate_normals}
    )
    print("made sand surface")

    # You can use all methods in optimesh:
    # points, cells = optimesh.cpt.fixed_point_uniform(
    # points, cells = optimesh.odt.fixed_point_uniform(
    # points_opt, cells_opt = optimesh.cvt.quasi_newton_uniform_blocks(
    #     conv_pts,
    #     cells,
    #     1.0e-2,
    #     10,
    #     verbose=True,
    #     implicit_surface=Sphere(),  # star_object(fname),
    #     # step_filename_format="out{:03d}.vtk"
    # )
    max_steps = 10
    q_check = call_back_test(max_num_steps=max_steps)

    points_opt, cells_opt = optimesh.cvt.quasi_newton_uniform_full(
        conv_pts,
        cells,
        1.0e-2,
        10,
        omega=0.9,
        verbose=True,
        callback=q_check.test_callback,
        uniform_density=True,
        implicit_surface=star_object(fname),
        implicit_surface_tol=1.0e-8
        # step_filename_format="out{:03d}.vtk"
    )

    tri_cells_opt = [("triangle", cells_opt)]

    mesh = meshio.write_points_cells("test_opt.vtk", points_opt, tri_cells_opt)
    print("optimized sand surface")


if __name__ == "__main__":
    main()

