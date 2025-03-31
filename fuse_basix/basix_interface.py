from fuse import *
import numpy as np
from basix import MapType, SobolevSpace, create_custom_element, PolysetType, CellType


basix_elements = { "interval": CellType.interval,
                   "triangle": CellType.triangle}

"""Fuse doesn't support L2Piola, doubleContravariantPiola, doubleCovariantPiola"""
basix_mapping = { "HCurl": MapType.covariantPiola,
                  "HDiv": MapType.contravariantPiola,
                  "H1": MapType.identity,
                  "L2": MapType.identity,
}

"""Fuse doesn't yet support H3, HDivDiv, HEin, HInf"""
basix_sobolev = {"L2": SobolevSpace.L2,
                 "H1": SobolevSpace.H1,
                 "HDiv": SobolevSpace.HDiv,
                 "HCurl": SobolevSpace.HCurl,
                 "H2": SobolevSpace.H2}

def compute_points_and_matrices(triple, ref_el):
    dofs = triple.generate()
    degree = triple.degree()

    spdim = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()

    min_ids = triple.cell.get_starter_ids()
    value_shape = triple.get_value_shape()
    if len(value_shape) > 1:
        value_size = value_shape[0]
    else:
        value_size = 1
    xs = [[np.zeros((0, spdim)) for entity in top[dim]] for dim in sorted(top) ]
    Ms = [[None for entity in top[dim]] for dim in sorted(top) ]

    entities = [(dim, entity) for dim in sorted(top) for entity in top[dim]]
    for entity in entities:
        dof_added = False
        dim = entity[0]
        for i in range(len(dofs)):
            if entity[1] == dofs[i].trace_entity.id - min_ids[dim]:
                dof_added = True
                converted = dofs[i].convert_to_fiat(ref_el, degree)
                pt_dict = converted.pt_dict
                dof_keys = list((pt_dict.keys()))
                dof_M = []
                for d in dof_keys:
                    wts = []
                    for pt in pt_dict[d]:
                        if len(value_shape) > 1:
                            wt = np.zeros_like(value_shape)
                            wt[pt[1]] = pt[0]
                        else:
                            wt = [pt[0]]
                        wts.append(wt)
                    dof_M.append(wts)
                derivs = converted.max_deriv_order
                if derivs == 0:
                    M = np.array([dof_M])
                else:
                    raise NotImplementedError("Derivatives need adding")
                xs[dim][entity[1]] = np.r_[xs[dim][entity[1]], np.array(transform_to_basix_cell(triple.cell, dof_keys))]

                if Ms[dim][entity[1]] is None:
                    Ms[dim][entity[1]] = M
                else:
                    Ms[dim][entity[1]] = np.r_[Ms[dim][entity[1]], M]

        if not dof_added:
            Ms[dim][entity[1]] = np.zeros((0, value_size, 0, 1))


    # remove when basix does this for me
    if len(xs) < 4:
        for _ in range(4 - len(xs)):
            xs.append([])
    if len(Ms) < 4:
        for _ in range(4 - len(Ms)):
            Ms.append([])

    return xs, Ms


def convert_to_basix_element(triple):
    ref_el = triple.cell.to_fiat()
    polyset = triple.spaces[0].to_ON_polynomial_set(ref_el)
    value_shape = list(triple.get_value_shape())
    x, M = compute_points_and_matrices(triple, ref_el)
    return create_custom_element(basix_elements[ref_el.cellname()],
                                 value_shape,
                                 polyset.coeffs,
                                 x,
                                 M,
                                 0,  # skip derivative for now
                                 basix_mapping[str(triple.spaces[1])],
                                 basix_sobolev[str(triple.spaces[1])],
                                 # if no dofs defined on boundary entities it is discontinuous
                                 sum(len(x[i]) for i in range(triple.cell.get_spatial_dimension())) == 0,
                                 triple.spaces[0].contains,
                                 triple.spaces[0].maxdegree,
                                 PolysetType.standard,  # don't yet support macro elements
                                 )


def ufc_triangle():
    vertices = []
    for i in range(3):
        vertices.append(Point(0))
    edges = []

    edges.append(Point(1, [vertices[0], vertices[1]], vertex_num=2))
    edges.append(Point(1, [vertices[1], vertices[2]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[2]], vertex_num=2))

    tri = Point(2, edges, vertex_num=3, variant="ufc", group=S1, edge_orientations={2: [1, 0]})
    return tri


basix_elements_fuserep = { "interval": Point(1, [Point(0), Point(0)], vertex_num=2, variant="ufc", group=S2),
                           "triangle": ufc_triangle()}


def transform_points(cell_a, cell_b, points):
    v_a = np.array(cell_a.vertices(return_coords=True))
    v_b = np.array(cell_b.vertices(return_coords=True))

    A = np.r_[v_a.T, np.ones((1, len(v_a)))]
    B = np.r_[v_b.T, np.ones((1, len(v_b)))]
    transform = B @ np.linalg.inv(A)

    res = (transform @ np.r_[np.array(points).T, np.ones((1, len(points)))])[:-1]
    res = [tuple(res.T[i]) for i in range(res.shape[1])]
    return res

def transform_to_basix_cell(fuse_cell, x):
    try:
        basix_cell = basix_elements_fuserep[fuse_cell.to_ufl().cellname()] 
    except KeyError:
        raise NotImplementedError(f"Fuse cell {fuse_cell.to_ufl().cellname()} doesn't have a Basix equivalent")
    
    return transform_points(fuse_cell, basix_cell, x)