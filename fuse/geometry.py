import numpy as np
import sympy as sp
import itertools


def construct_attach_2d(a, b, c, d):
    """
    Compute polynomial attachment in x based on two points (a,b) and (c,d)

    :param: a,b,c,d: two points (a,b) and (c,d)
    """
    x = sp.Symbol("x")
    return [((c-a)/2)*(x+1) + a, ((d-b)/2)*(x+1) + b]


def construct_attach_3d(res):
    """
    Convert matrix of coefficients into a vector of polynomials in x and y

    :param: res: matrix of coefficients
    """
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    xy = sp.Matrix([1, x, y])
    return (xy.T * res)


def compute_equilateral_verts(d, n):
    """
    Construct default cell vertices

    :param: d: dimension of cell
    :param: n: number of vertices
    """
    if d == 1:
        return np.array([[-1], [1]])
    elif d == 2:
        source = np.array([0, 1])
        rot_coords = [source for i in range(0, n)]

        rot_mat = np.array([[np.cos((2*np.pi)/n), -np.sin((2*np.pi)/n)], [np.sin((2*np.pi)/n), np.cos((2*np.pi)/n)]])
        for i in range(1, n):
            rot_coords[i] = np.matmul(rot_mat, rot_coords[i-1])
        xdiff, ydiff = (rot_coords[0][0] - rot_coords[1][0],
                        rot_coords[0][1] - rot_coords[1][1])
        scale = 2 / np.sqrt(xdiff**2 + ydiff**2)
        scaled_coords = np.array([[scale*x, scale*y] for (x, y) in rot_coords])
        return scaled_coords
    elif d == 3:
        if n == 4:
            A = [-1, 1, -1]
            B = [1, -1, -1]
            C = [1, 1, 1]
            D = [-1, -1, 1]
            coords = [A, B, C, D]
            face1 = np.array([A, D, C])
            face2 = np.array([A, B, D])
            face3 = np.array([A, C, B])
            face4 = np.array([B, D, C])
            faces = [face1, face2, face3, face4]
        elif n == 8:
            coords = []
            faces = [[] for i in range(6)]
            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [-1, 1]:
                        coords.append([i, j, k])

            for j in [-1, 1]:
                for k in [-1, 1]:
                    faces[0].append([1, j, k])
                    faces[1].append([-1, j, k])
                    faces[2].append([j, 1, k])
                    faces[3].append([j, -1, k])
                    faces[4].append([j, k, 1])
                    faces[5].append([j, k, -1])

        else:
            raise ValueError("Polyhedron with {} vertices not supported".format(n))

        xdiff, ydiff, zdiff = (coords[0][0] - coords[1][0],
                               coords[0][1] - coords[1][1],
                               coords[0][2] - coords[1][2])
        scale = 2 / np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
        scaled_coords = np.array([[scale*x, scale*y, scale*z] for (x, y, z) in coords])
        scaled_faces = np.array([[[scale*x, scale*y, scale*z] for (x, y, z) in face] for face in faces])

        return scaled_coords, scaled_faces
    else:
        raise ValueError("Dimension {} not supported".format(d))


def compute_ufc_verts(d, n):
    """
    Construct UFC cell vertices

    :param: d: dimension of cell
    :param: n: number of vertices

    sorts the list for a consistent ordering.
    """
    if d + 1 == n:
        values = [1] + [0 for _ in range(d)]
    elif 2**d == n:
        values = [1 for _ in range(d)] + [0 for _ in range(d)]
    else:
        raise NotImplementedError(f"Not able to construct cell in {d} dimensions with {n} vertices.")

    # remove duplicates, sort starting with first element, convert to np float
    vertices = np.array(sorted(list(set(itertools.permutations(values, d))), key=lambda x: x[::-1]), dtype=np.float64)

    if d == 3:
        raise NotImplementedError("Need to construct UFC faces in 3D")
        if n == 4:
            faces = set(itertools.permutations(vertices, 3))
            print(faces)
        elif n == 8:
            faces = set(itertools.permutations(vertices, 4))
        return vertices
    return vertices


coord_variants = {"equilateral": compute_equilateral_verts,
                  "ufc": compute_ufc_verts}


def compute_attachment_1d(n, variant="equilateral"):
    """
    Constructs sympy functions for attaching the points of
    n vertices and coordinates determined by variant.

    :param: n: number of vertices
    :param: variant: vertex type"""
    coords = coord_variants[variant](1, n)
    return [sp.sympify(tuple(c)) for c in coords]


def compute_attachment_2d(n, variant="equilateral"):
    """
    Constructs sympy functions for attaching the edges of a
    polygon with n vertices and coordinates determined by variant.

    :param: n: number of vertices
    :param: variant: vertex type"""
    coords = coord_variants[variant](2, n)
    attachments = []

    for i in range(n):
        a, b = coords[i]
        c, d = coords[(i + 1) % n]

        attachments.append(construct_attach_2d(a, b, c, d))
    return attachments


def compute_attachment_3d(n, variant="equilateral"):
    coords, faces = compute_equilateral_verts(3, n)
    coords_2d = np.c_[np.ones(len(faces[0])), compute_equilateral_verts(2, len(faces[0]))]
    res = []
    attachments = []

    for i in range(len(faces)):
        res = np.linalg.solve(coords_2d, faces[i])

        res_fn = construct_attach_3d(res)

        assert np.allclose(np.array(res_fn.subs({"x": coords_2d[0][1], "y": coords_2d[0][2]})).astype(np.float64), faces[i][0])
        assert np.allclose(np.array(res_fn.subs({"x": coords_2d[1][1], "y": coords_2d[1][2]})).astype(np.float64), faces[i][1])
        assert np.allclose(np.array(res_fn.subs({"x": coords_2d[2][1], "y": coords_2d[2][2]})).astype(np.float64), faces[i][2])

        attachments.append(construct_attach_3d(res))
    return attachments
