from fuse import *
import math
import numpy as np
import sympy as sp
from recursivenodes import recursive_nodes


def convert_to_generation(coords, verts):
    """Reduces a full list of cartesian coordinates to only those required for generation,
       and divides them into groups """
    coords_grps = {}
    n = len(coords)
    for c in coords:
        try:
            grp = identify_generation_group(c, verts, bary=False)
            if grp in coords_grps.keys():
                coords_grps[grp] += [c]
            else:
                coords_grps[grp] = [c]
        except ValueError:
            pass

    assert n == sum([len(coords_grps[grp])*grp.size() for grp in coords_grps.keys()])
    return coords_grps


def identify_generation_group(b_coord, verts, bary=True):
    """Identify the correct generation group from a (optionally) barycentric coordinate on a triangle or interval.

        Assumes the coordinate lies in the lower left region of the triangle, defined by the sorted vertices."""
    if bary:
        # respect vertex order to compute barycentric coord
        assert len(b_coord) == len(verts)
        c = sum(b_coord[i]*np.array(verts)[i] for i in range(len(verts)))
    else:
        c = b_coord
    # ensure consistent vertex order to check points in given region
    verts = np.array(verts)
    inds = np.lexsort(tuple(verts[:, i] for i in range(len(verts) - 2, -1, -1)))  # lex sort prioritises last row, so reverse order
    verts = verts[inds]

    center = tuple(sum([v[i] for v in verts])/len(verts) for i in range(len(verts[0])))
    if np.allclose(center, c):
        return S1

    if len(verts) == 2:
        return S2
    if len(verts) == 3:
        midpoint1 = ((verts[1][0] + verts[0][0])/2, (verts[1][1] + verts[0][1])/2)
        midpoint2 = ((verts[0][0] + verts[2][0])/2, (verts[0][1] + verts[2][1])/2)
        cond1 = lambda coord: check_multiple(coord, verts[0]) and check_below_line(verts[2], midpoint1, coord) >= 0
        cond2 = lambda coord: (check_multiple(coord, midpoint2) and check_below_line(midpoint1, (0, 0), coord) >= 0)
        if cond1(c):
            return C3
        elif cond2(c):  # cond3(c) or
            return diff_C3
        elif check_below_line(verts[0], (0, 0), c) == 1 and check_below_line(midpoint2, (0, 0), c) == 1:
            return S3
    if len(verts) == 4:
        midpoint1 = (verts[1] + verts[0])/2
        midpoint2 = (verts[0] + verts[2])/2
        midpoint3 = (verts[1] + verts[3])/2
        midpoint4 = (verts[2] + verts[3])/2
        plane1 = lambda coord: check_below_plane(verts[2], verts[3], midpoint1, coord) >= 0
        plane2 = lambda coord: check_below_plane(verts[1], verts[3], midpoint2, coord) >= 0
        plane3 = lambda coord: check_below_plane(verts[0], verts[2], midpoint3, coord) >= 0
        plane4 = lambda coord: check_below_plane(verts[0], verts[1], midpoint4, coord) >= 0
        if check_multiple(c, verts[0]) and plane1(c) and plane2(c):
            return C4
        # add diff C4 ?? need to work out if needed
        elif plane1(c) and plane2(c) and plane3(c) and plane4(c):
            return tet_faces
    raise ValueError("Group not identified")


def check_below_line(x_0, x_1, coord, fix_order=True):
    """
    Checks position of coord in relation to the segment defined by x_0 and x_1.

    Fixes order of input coords such that line segment is oriented left to right (or down to up),
    returning 1 if coord is below or to the left.
    """
    if fix_order:
        if x_1[0] < x_0[0] and not np.allclose(x_1[0], x_0[0]):
            temp = x_0
            x_0 = x_1
            x_1 = temp
        elif np.allclose(x_1[0], x_0[0]) and x_1[1] > x_0[1]:
            temp = x_0
            x_0 = x_1
            x_1 = temp
    coord = np.array(coord)
    v_0 = np.array(x_1) - np.array(x_0)
    n = np.matmul(np.array([[0, -1], [1, 0]]), v_0)
    eq = lambda x: np.dot(n, x - x_0)

    if np.allclose(eq(coord), 0):
        return 0
    elif eq(coord) < 0:
        return 1
    else:
        return -1


def check_below_plane(x_0, x_1, x_2, coord):
    v_0 = np.array(x_1) - np.array(x_0)
    v_1 = np.array(x_2) - np.array(x_0)
    n = np.cross(v_0, v_1)
    eq = lambda x: np.dot(n, x - x_0)

    if np.allclose(eq(coord), 0):
        return 0
    elif eq(coord[0]) < 0:
        return 1
    else:
        return -1


def check_multiple(coord_1, coord_2):
    if len(coord_1) == 2:
        return np.allclose(check_below_line(coord_2, (0, 0), coord_1), 0)
    div = coord_1 / coord_2
    return np.allclose(div, div[0])


def lagrange_barycentric_basis(dim, verts, deg):
    symbols = []
    for i in range(dim + 1):
        symbols += [sp.Symbol(f"s_{i}")]

    if dim == 2:
        # hack to ensure (-1, -np.sqrt(3)/3) is the first vertex
        verts = verts[1:] + [verts[0]]

    # Construct multiindices of the generation basis functions
    acc_indices = [tuple()]
    multiindices = []
    ext = deg + 1
    for d in range(dim + 1):
        temp = []
        for idx in acc_indices:
            if len(idx) > 0:
                ext = idx[-1] + 1
            for i in range(0, ext):
                temp += [idx + (i,)]
        acc_indices = temp
    for idx in acc_indices:
        if sum(idx) == deg:
            multiindices += [idx]

    scale = 1 if deg == 0 else deg
    grps = [identify_generation_group(tuple(i / scale for i in idx), verts) for idx in multiindices]
    const = lambda idx: math.factorial(deg) / math.prod(math.factorial(i) for i in idx)
    fns = [const(idx)*math.prod(s**i for s, i in zip(symbols, idx))for idx in multiindices]
    return fns, grps, symbols


def proxy_field_1_form(cell, deg, rot=False):
    symbols = []
    for i in range(cell.dimension + 1):
        symbols += [sp.Symbol(f"s_{i}")]
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    v_0 = np.array(cell.ordered_vertex_coords()[0])
    bvs = np.array(cell.basis_vectors(norm=False))
    res = np.matmul(np.linalg.inv(bvs.T), np.array((x, y) - v_0))
    ls = (1 - sum(res),) + tuple(res[i] for i in range(len(res)))
    dl = []
    for l in ls:
        dl += [sp.Matrix((sp.diff(l, x), sp.diff(l, y)))]
    bfs = [sp.Matrix(symbols[i]*dl[j] - symbols[j]*dl[i]) for (i, j) in [(0, 1), (0, 2), (1, 2)]]
    if rot:
        perp = lambda x: np.array([[0, -1], [1, 0]]) @ x
        bfs = [perp(bf) for bf in bfs]
    return bfs, [diff_C3], symbols


def vector_facet_fns(cell, deg, rot=False):
    """
    Returns vector valued basis functions of a given degree over a given cell, by default returning
    Nedelec basis functions, and returning Raviart Thomas if rot=True
    """
    edge = cell.edges()[0]
    nd_basis_funcs, nd_grps, symbols = proxy_field_1_form(cell, deg, rot)
    basis_funcs, groups, symbols = lagrange_barycentric_basis(edge.dimension, edge.ordered_vertex_coords(), deg - 1)
    dofs = []
    for nd_bf, nd_grp in zip(nd_basis_funcs, nd_grps):
        for bf, grp in zip(basis_funcs, groups):
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(nd_bf*bf, symbols=symbols))]
            if grp.size() > 3:
                breakpoint()
            dofs += [DOFGenerator(xs, nd_grp*grp, S1)]

    basis_funcs, groups, symbols = lagrange_barycentric_basis(cell.dimension, cell.ordered_vertex_coords(), deg - 2)
    for bf, grp in zip(basis_funcs, groups):
        v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
        v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
        v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))
        xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*bf*(v_0 - v_2)/2, symbols=symbols))]
        if grp.size() > 3:
            dofs += [DOFGenerator(xs, grp, S1)]
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*bf*(v_0 - v_1)/2, symbols=symbols))]
            dofs += [DOFGenerator(xs, grp, S1)]
        else:
            dofs += [DOFGenerator(xs, S2*grp, S1)]
    return dofs


def lagrange_facet_fns(shp, cell, deg):
    dofs = []
    basis_funcs, groups, symbols = lagrange_barycentric_basis(cell.dimension, cell.ordered_vertex_coords(), deg)
    for bf, grp in zip(basis_funcs, groups):
        if shp == 1:
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf, symbols=symbols))]
            dofs += [DOFGenerator(xs, grp, S2)]
        elif shp == 2:
            v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
            v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
            v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_2)/2, symbols=symbols))]
            if grp.size() > 3:
                dofs += [DOFGenerator(xs, grp, S1)]
                xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_1)/2, symbols=symbols))]
                dofs += [DOFGenerator(xs, grp, S1)]
            else:
                dofs += [DOFGenerator(xs, S2*grp, S1)]
        else:
            raise NotImplementedError("Facet functions on facets with dimension greater than 2 not supported")
    return dofs


def lagrange_facet_pts(cell, deg):
    from recursivenodes.nodes import _recursive, _decode_family
    from FIAT.reference_element import multiindex_equal
    get_pt = lambda alpha: np.dot(_recursive(cell.dimension, deg, alpha, _decode_family("lgl")), np.array(cell.vertices(return_coords=True)))
    interior_coords = [tuple(pt) for pt in list(map(get_pt, multiindex_equal(cell.dimension + 1, deg, 1)))]
    coords_grps = convert_to_generation(interior_coords, cell.ordered_vertex_coords())

    int_dofs = []
    for grp in coords_grps.keys():
        dofs = [DOF(DeltaPairing(), PointKernel(c)) for c in coords_grps[grp]]
        if len(dofs) > 0:
            int_dofs += [DOFGenerator([d], grp, S1) for d in dofs]
    return int_dofs


def construct_tri_cgN(deg):
    cell = polygon(3)
    vert = cell.vertices()[0]
    edge = cell.edges()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))
    v_xs = [immerse(cell, dg0, TrH1)]
    v_dofs = [DOFGenerator(v_xs, C3, S1)]

    points = recursive_nodes(1, deg, domain="equilateral")[1:-1].flatten()

    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    sym_dofs = [DOFGenerator([pt], S2, S1) for pt in sym_points]
    if 0 in points:
        centre_dof = [DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)]
    else:
        centre_dof = []
    edge_dg0 = ElementTriple(edge, (Pk, CellL2, C0), sym_dofs + centre_dof)
    edge_xs = [immerse(cell, edge_dg0, TrH1)]
    edge_dofs = [DOFGenerator(edge_xs, C3, S1)]

    int_dofs = lagrange_facet_pts(cell, deg)
    return ElementTriple(cell, (Pk, CellL2, C0), v_dofs + edge_dofs + int_dofs)


def construct_tet_cgN(deg):
    cell = make_tetrahedron()
    vert = cell.vertices()[0]
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))
    v_xs = [immerse(cell, dg0, TrH1)]
    v_dofs = [DOFGenerator(v_xs, C4, S1)]

    points = recursive_nodes(1, deg, domain="equilateral")[1:-1].flatten()

    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    sym_dofs = [DOFGenerator([pt], S2, S1) for pt in sym_points]
    if 0 in points:
        centre_dof = [DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)]
    else:
        centre_dof = []
    edge_dg0 = ElementTriple(edge, (Pk, CellL2, C0), sym_dofs + centre_dof)
    edge_xs = [immerse(cell, edge_dg0, TrH1)]
    edge_dofs = [DOFGenerator(edge_xs, tet_edges, S1)]

    face_dofs = lagrange_facet_pts(face, deg)
    face_trp = ElementTriple(face, (Pk, CellL2, C0), face_dofs)
    face_xs = [immerse(cell, face_trp, TrH1)]
    face_dofs = [DOFGenerator(face_xs, tet_faces, S1)]

    int_dofs = lagrange_facet_pts(cell, deg)

    cg = ElementTriple(cell, (Pk, CellL2, C0), v_dofs + edge_dofs + face_dofs + int_dofs)
    assert len(cg.generate()) == (deg + 1)*(deg + 2)*(deg + 3)/6
    return cg


def construct_tri_ndN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    dofs = lagrange_facet_fns(1, edge, deg - 1)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), dofs)
    xs = [immerse(cell, int_ned1, TrHCurl)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    center_dofs = lagrange_facet_fns(2, cell, deg - 2)

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

    ned = ElementTriple(cell, (nd, CellHCurl, C0), tri_dofs + center_dofs)
    return ned


def construct_tri_ndN_2(deg):
    cell = polygon(3)
    edge = cell.edges()[0]
    verts = cell.vertices(return_coords=True)
    verts.reverse()
    verts = [verts[0], verts[2], verts[1]]

    dofs = lagrange_facet_fns(1, edge, deg)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), dofs)
    xs = [immerse(cell, int_ned1, TrHCurl)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    dofs = vector_facet_fns(cell, deg - 1, rot=True)

    vec_Pk = PolynomialSpace(deg, set_shape=True)

    ned = ElementTriple(cell, (vec_Pk, CellHCurl, C0), tri_dofs + dofs)
    assert len(ned.generate()) == (deg + 1)*(deg + 2)
    return ned


def construct_tri_rtN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    x = sp.Symbol("x")
    y = sp.Symbol("y")

    dofs = lagrange_facet_fns(1, edge, deg - 1)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHDiv, C0), dofs)
    xs = [immerse(cell, int_ned1, TrHDiv)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    center_dofs = lagrange_facet_fns(2, cell, deg - 2)

    x = sp.Symbol("x")
    y = sp.Symbol("y")

    M = sp.Matrix([[x, y]])
    vec_Pd = PolynomialSpace(deg - 1, set_shape=True)
    Pd = PolynomialSpace(deg - 1)
    rt = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    rt = ElementTriple(cell, (rt, CellHDiv, C0), tri_dofs + center_dofs)
    return rt


def construct_tri_bdmN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    dofs = lagrange_facet_fns(1, edge, deg)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHDiv, C0), dofs)
    xs = [immerse(cell, int_ned1, TrHDiv)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    center_dofs = vector_facet_fns(cell, deg - 1)

    vec_Pd = PolynomialSpace(deg, set_shape=True)

    ned = ElementTriple(cell, (vec_Pd, CellHDiv, C0), tri_dofs + center_dofs)
    assert len(ned.generate()) == (deg + 1)*(deg + 2)
    return ned


def construct_tri_dgN(deg):
    return construct_tri_dgNminus(deg + 1)


def construct_tri_dgNminus(deg):
    cell = polygon(3)
    Pk = PolynomialSpace(deg)
    int_dofs = lagrange_facet_pts(cell, deg + 3)
    return ElementTriple(cell, (Pk, CellL2, C0), int_dofs)


# column: dimension: form number
constructors = {
    0: {
        2: {
            0: construct_tri_cgN,
            1: construct_tri_ndN,
            2: construct_tri_rtN,
            3: construct_tri_dgNminus,
        },
        3: {
            0: construct_tet_cgN,
        },
    },
    1: {
        2: {
            0: construct_tri_cgN,
            1: construct_tri_ndN_2,
            2: construct_tri_bdmN,
            3: construct_tri_dgN,
        },
    },
}


def periodic_table(col, dim, k, deg):
    return constructors[col][dim][k](deg)
