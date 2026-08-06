from fuse import *
import numpy as np
# import matplotlib.pyplot as plt


triangle_nums = [sum(range(i)) for i in range(2, 20)]


def triangle_coords(n, verts=[(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)], scale=0.8):
    assert n in triangle_nums
    if n == 1:
        return [(0, 0)]
    on_edge = triangle_nums.index(n) - 1
    sub_verts = [(scale*x, scale*y) for (x, y) in verts]
    if on_edge > 0:
        ratio = 1 / (on_edge + 1)
        for i in range(on_edge):
            sub_verts += [(sub_verts[0][0] + ratio*(i+1)*(sub_verts[1][0] - sub_verts[0][0]), sub_verts[0][1] + ratio*(i+1)*(sub_verts[1][1] - sub_verts[0][1]))]
            sub_verts += [(sub_verts[0][0] + ratio*(i+1)*(sub_verts[2][0] - sub_verts[0][0]), sub_verts[0][1] + ratio*(i+1)*(sub_verts[2][1] - sub_verts[0][1]))]
            sub_verts += [(sub_verts[2][0] + ratio*(i+1)*(sub_verts[1][0] - sub_verts[2][0]), sub_verts[2][1] + ratio*(i+1)*(sub_verts[1][1] - sub_verts[2][1]))]
        if (n - len(sub_verts)) > 0:
            int_scale = (sub_verts[0][0] + ratio*3*(sub_verts[1][0] - sub_verts[0][0])) / sub_verts[0][0]
            interior = triangle_coords((n - len(sub_verts)), sub_verts[0:3], int_scale)
            sub_verts += interior
    assert len(sub_verts) == n
    return sub_verts


def convert_to_generation(coords, verts=[(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)]):
    coords_S1 = []
    coords_C3 = []
    coords_S3 = []
    n = len(coords)
    center = (sum([x for (x, y) in verts])/len(verts), sum([y for (x, y) in verts])/len(verts))
    if center in coords:
        coords_S1 += [center]
        coords.remove(center)
    divide_1 = ((verts[0][0] + verts[1][0])/2, (verts[0][1] + verts[1][1])/2)
    divide_2 = ((verts[0][0] + verts[2][0])/2, (verts[0][1] + verts[2][1])/2)
    for coord in coords:
        if check_multiple(coord, verts[0]) or (check_multiple(coord, divide_1) and check_below_line(divide_2, (0, 0), coord) <= 0) or (check_multiple(coord, divide_2) and check_below_line(divide_1, (0, 0), coord) <= 0):
            coords_C3 += [coord]
        elif check_below_line(verts[0], (0, 0), coord) == -1 and check_below_line(divide_2, (0, 0), coord) == -1:
            coords_S3 += [coord]
    assert n == len(coords_S1) + len(coords_S3)*6 + len(coords_C3)*3
    return coords_S1, coords_C3, coords_S3


def check_below_line(seg_1, seg_2, coord):
    if seg_1[0] - seg_2[0] == 0:
        if coord[0] == seg_1[0]:
            return 0
        elif coord[0] > seg_1[0]:
            return 1
        else:
            return -1

    if seg_1[1] - seg_2[1] == 0:
        if coord[1] == seg_1[1]:
            return 0
        elif coord[1] > seg_1[1]:
            return 1
        else:
            return -1

    m = (seg_1[1] - seg_2[1])/(seg_1[0] - seg_2[0])
    c = seg_1[1] - seg_1[0]*m
    eq = lambda x: m*x + c

    if eq(coord[0]) == coord[1]:
        return 0
    elif eq(coord[0]) < coord[1]:
        return 1
    else:
        return -1


def check_multiple(coord_1, coord_2):
    return check_below_line(coord_2, (0, 0), coord_1) == 0


def CR_n(cell, deg):
    points = np.polynomial.legendre.leggauss(deg)[0]
    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    if 0 in points:
        edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0, Fid), [DOFGenerator(sym_points, S2, S1),
                                                                                   DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)])
    else:
        edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0, Fid), [DOFGenerator(sym_points, S2, S1)])
    edge_xs = [immerse(cell, edge_dg0, TrH1)]

    interior_coords = triangle_coords(triangle_nums[deg - 3])
    s1, c3, s3 = convert_to_generation(interior_coords)

    s1 = [DOF(DeltaPairing(), PointKernel(c)) for c in s1]
    c3 = [DOF(DeltaPairing(), PointKernel(c)) for c in c3]
    s3 = [DOF(DeltaPairing(), PointKernel(c)) for c in s3]

    return ElementTriple(cell, (Pk, C0, Fid), [DOFGenerator(edge_xs, C3, S1), DOFGenerator(s1, S1, S1), DOFGenerator(c3, C3, S1), DOFGenerator(s3, S3, S1)])


# coords = triangle_coords(21)
# edge_coords = [(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)]

# ax = plt.gca()
# ax.scatter([e[0] for e in coords], [e[1] for e in coords], color="grey")
# ax.scatter([e[0] for e in edge_coords], [e[1] for e in edge_coords], color="grey")
# ax.figure.savefig("triangle.png")

# s1, c3, s3 = convert_to_generation(coords)
# gen_coords = s1+c3+s3
# ax = plt.gca()
# ax.scatter([e[0] for e in s1], [e[1] for e in s1], color="black")
# ax.scatter([e[0] for e in c3], [e[1] for e in c3], color="green")
# ax.scatter([e[0] for e in s3], [e[1] for e in s3], color="blue")
# # ax.scatter([e[0] for e in edge_coords], [e[1] for e in edge_coords], color="blue")
# # ax.figure.savefig("triangle_gen.png")
# crn = CR_n(polygon(3), 5)
# print(len(crn.generate()))
# for d in crn.generate():
#     print(d)
