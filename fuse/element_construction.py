from fuse import *
import math
import numpy as np
import sympy as sp
from recursivenodes import recursive_nodes
import itertools
from functools import reduce
from operator import mul


def convert_to_generation(coords, verts, return_idx=False):
    """Reduces a full list of cartesian coordinates to only those required for generation,
       and divides them into groups
       return idx argument returns generators as index into list. without this, the coordinate is returned"""
    n = len(coords)
    verts = np.array(verts)
    coords_grps = group_with_mappings(coords, verts, return_idx)
    assert n == sum([len(coords_grps[grp])*grp.size() for grp in coords_grps.keys()])
    return coords_grps


def barycentric_coords(p, verts):
    """
    Compute barycentric coordinates of point p wrt listed of verts
    """
    A = np.vstack((verts.T, np.ones(len(verts))))
    b = np.append(p, 1.0)
    return np.linalg.solve(A, b)


def from_barycentric(lmbda, verts):
    return np.dot(lmbda, verts)


def all_permutations():
    return list(itertools.permutations(range(4)))  # 24 permutations


def points_close(p, q, tol=1e-6):
    return np.linalg.norm(p - q) < tol


# ---------- main grouping ----------

def find_permutation(lam_i, lam_j, tol=1e-6):
    """
    Find permutation sigma such that lam_j ≈ lam_i permuted by sigma.
    Returns permutation as a tuple, or None.
    """
    perms = []
    for perm in itertools.permutations(range(len(lam_i))):
        if np.allclose(lam_i, lam_j[list(perm)], atol=tol):
            perms += [perm]
    if len(perms) > 1:
        for p in perms:
            if Permutation(p).is_even:
                return p
    return perms[0]
    return None


def group_with_mappings(points, verts, return_idx=False, tol=1e-6):
    points = [np.array(p) for p in points]

    # compute barycentric coordinates
    bary = [barycentric_coords(p, verts) for p in points]

    # first group using fast method
    raw_groups = group_by_symmetry(points, verts, tol)
    result = {}

    for group in raw_groups:
        base = group[0]
        lam_base = bary[base]

        perm_list = []
        for j in group:
            lam_j = bary[j]

            perm = find_permutation(lam_base, lam_j, tol)

            if perm is None:
                raise ValueError("Failed to find permutation")

            perm_list += [Permutation(perm)]
        perm_group = PermutationSetRepresentation(perm_list)
        if return_idx:
            result[perm_group] = [base]
        else:
            result[perm_group] = [tuple(points[base])]

    return result


def group_by_symmetry(points, verts, tol=1e-6):
    """
    Group points into symmetry orbits using sorted barycentric coordinates.
    """
    points = [np.array(p) for p in points]

    groups = {}
    for i, p in enumerate(points):
        lam = barycentric_coords(p, verts)

        # numerical cleanup and normalise
        lam[np.abs(lam) < tol] = 0.0
        lam = lam / np.sum(lam)

        # canonical key = sorted barycentric coords
        key = tuple(np.round(np.sort(lam), decimals=6))

        if key not in groups:
            groups[key] = []

        groups[key].append(i)

    return list(groups.values())


def multiindices(k, n):
    if n == 1:
        yield (k,)
    else:
        for i in range(k + 1):
            for tail in multiindices(k - i, n - 1):
                yield (i,) + tail


def lagrange_barycentric_basis(dim, verts, deg):
    symbols = []
    for i in range(dim + 1):
        symbols += [sp.Symbol(f"s_{i}")]
    verts = np.array(verts)

    scale = 1 if deg == 0 else deg
    new_mul = list(multiindices(deg, dim + 1))
    coords = [sum((i / scale)*verts[j] for j, i in enumerate(idx)) for idx in new_mul]
    coords_grps = convert_to_generation(coords, verts, return_idx=True)

    grps = [grp for grp in coords_grps.keys()]
    const = lambda idx: math.factorial(deg) / math.prod(math.factorial(i) for i in idx)
    fns = [const(new_mul[idx])*math.prod(s**i for s, i in zip(symbols, new_mul[idx])) for idxs in coords_grps.values() for idx in idxs]
    return fns, grps, symbols


def proxy_field_bfs(cell, deg, rot=False):
    symbols = []
    coords = []
    for i in range(cell.dimension + 1):
        symbols += [sp.Symbol(f"s_{i}")]
        if i < cell.dimension:
            coords += [sp.Symbol(f"s_{i}")]
    v_0 = np.array(cell.ordered_vertex_coords()[0])
    bvs = np.array(cell.basis_vectors(norm=False))
    res = np.matmul(np.linalg.inv(bvs.T), np.array(coords - v_0))
    ls = (1 - sum(res),) + tuple(res[i] for i in range(len(res)))
    dl = []
    for l in ls:
        dl += [sp.Matrix([sp.diff(l, x) for x in coords])]
    bfs = [sp.Matrix(symbols[i]*dl[j] - symbols[j]*dl[i]) for (i, j) in [(0, 1), (0, 2), (1, 2)]]
    if cell.dimension == 2:
        grp = [diff_C3]
    else:
        grp = [tet_edges]
    if rot:
        # assert cell.dimension <= 2
        if cell.dimension == 2:
            perp = lambda x: np.array([[0, -1], [1, 0]]) @ x
            bfs = [perp(bf) for bf in bfs]
            grp = [C3]
        else:
            bfs = [2*sp.Matrix(symbols[i]*dl[j].cross(dl[k]) - symbols[j]*dl[i].cross(dl[k]) + symbols[k]*dl[i].cross(dl[j])) for i, j, k in [[0, 1, 2]]]
            grp = [tet_faces]

    return bfs, grp, symbols


def vector_basis_fns(cell, deg, rot=False):
    """
    Returns vector valued basis functions of a given degree over a given cell, by default returning
    Nedelec basis functions, and returning Raviart Thomas if rot=True
    """
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]
    nd_basis_funcs, nd_grps, symbols = proxy_field_bfs(cell, deg, rot)
    basis_funcs, groups, symbols = lagrange_barycentric_basis(edge.dimension, edge.ordered_vertex_coords(), deg - 1)
    if cell.dimension == 3 and rot:
        basis_funcs, groups, symbols = lagrange_barycentric_basis(face.dimension, face.ordered_vertex_coords(), deg - 1)
    dofs = []
    for nd_bf, nd_grp in zip(nd_basis_funcs, nd_grps):
        for bf, grp in zip(basis_funcs, groups):
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(nd_bf*bf, symbols=symbols))]
            if grp.size() != 1:
                if cell.dimension == 3 and grp.size() == 2:
                    dofs += [DOFGenerator(xs, nd_grp*tet_C2, S1)]
                else:
                    dofs += [DOFGenerator(xs, nd_grp*grp, S1)]
            else:
                dofs += [DOFGenerator(xs, nd_grp, S1)]

    interior_deg = deg - 2

    if cell.dimension == 3 and interior_deg >= 0 and not rot:
        face = cell.d_entities(2)[0]
        face_dofs = vector_basis_fns(face, deg)
        if len(face_dofs) > 0:
            face_dofs = face_dofs[-1]

            def immersed(face, pt, o):
                basis = np.array(face.orient(o).basis_vectors()).T
                basis_coeffs = np.matmul(np.linalg.inv(basis), np.array(pt))
                J = np.array(cell.basis_vectors(entity=face)).T
                return np.matmul(J, basis_coeffs)
            original_kernel = face_dofs.x[0].kernel
            extra_sym = sp.Symbol("s_3")
            symbols = original_kernel.syms + [extra_sym]
            new_dofs = []
            for f in cell.d_entities(2):
                bary_verts = np.rint(np.array(cell.cartesian_to_barycentric([cell.get_node(v, return_coords=True) for v in f.ordered_vertices()]))).astype(np.int64)
                new_kernel_fn = [poly.subs({o_sym: sum(symbols[j] for j in range(len(b)) if b[j] == 1) for o_sym, b in zip(original_kernel.syms, bary_verts)}) for poly in original_kernel.fn]
                kernels = [type(original_kernel)(immersed(f, new_kernel_fn, o), symbols=symbols) for o in face_dofs.g1.add_cell(f).members()]
                new_dofs += [DOF(face_dofs.x[0].pairing, kernel) for kernel in kernels]

            # grp = face_dofs.g1
            # if grp.size() == 2:
            #     # grp = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]), Permutation([1, 3, 2, 0]), Permutation([3, 0, 2, 1]),
            #     #                                     Permutation([2, 1, 0, 3]), Permutation([0, 2, 3, 1]), Permutation([2, 3, 0, 1]), Permutation([3, 0, 1, 2])])
            #     grp = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]), Permutation([1, 3, 2, 0]), Permutation([3, 0, 2, 1]),
            #                                         Permutation([3, 1, 2, 0]), Permutation([1, 2, 0, 3]), Permutation([1, 3, 0, 2]), Permutation([1, 0, 2, 3])])
            #     breakpoint()
            # else:
            #     grp = tet_faces*grp
            dofs += [DOFGenerator(new_dofs, S1, S1)]
            interior_deg = deg - 3

    basis_funcs, groups, symbols = lagrange_barycentric_basis(cell.dimension, cell.ordered_vertex_coords(), interior_deg)
    for bf, grp in zip(basis_funcs, groups):
        v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
        v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
        v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))

        xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*bf*(v_0 - v_2)/2, symbols=symbols))]
        if cell.dimension == 3:
            if grp.size() == 2:
                grp = tet_C2
            v_3 = np.array(cell.get_node(cell.ordered_vertices()[3], return_coords=True))
            dofs += [DOFGenerator(xs, grp, S1)]
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*bf*(v_0 - v_1)/2, symbols=symbols))]
            dofs += [DOFGenerator(xs, grp, S1)]
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*bf*(v_0 - v_3)/2, symbols=symbols))]
            dofs += [DOFGenerator(xs, grp, S1)]
        else:
            if grp.size() > 3:
                dofs += [DOFGenerator(xs, grp, S1)]
                xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*bf*(v_0 - v_1)/2, symbols=symbols))]
                dofs += [DOFGenerator(xs, grp, S1)]
            else:
                dofs += [DOFGenerator(xs, S2*grp, S1)]
    return dofs


def lagrange_facet_fns(cell, deg, interior=False, vector=False):
    dofs = []
    if interior:
        g2 = S1
    elif not vector:
        # This is probably not strictly right - true for ND and RT nodes on the face they are associated with
        g2 = S2
    else:
        g2 = cell.group
    basis_funcs, groups, symbols = lagrange_barycentric_basis(cell.dimension, cell.ordered_vertex_coords(), deg)
    for bf, grp in zip(basis_funcs, groups):
        if not vector:
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf, symbols=symbols))]
            dofs += [DOFGenerator(xs, grp, g2)]
        else:
            v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
            v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
            v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_2)/2, symbols=symbols))]
            if cell.dimension == 3:
                # if grp.size() > 3:
                v_3 = np.array(cell.get_node(cell.ordered_vertices()[3], return_coords=True))
                dofs += [DOFGenerator(xs, grp, g2)]
                xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_1)/2, symbols=symbols))]
                dofs += [DOFGenerator(xs, grp, g2)]
                xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_3)/2, symbols=symbols))]
                dofs += [DOFGenerator(xs, grp, g2)]
                # else:
                #     dofs += [DOFGenerator(xs, C3*grp, g2)]
            else:
                # raise NotImplementedError("basis group for tets")
                if grp.size() > 3:
                    dofs += [DOFGenerator(xs, grp, g2)]
                    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_1)/2, symbols=symbols))]
                    dofs += [DOFGenerator(xs, grp, g2)]
                else:

                    dofs += [DOFGenerator(xs, S2*grp, g2)]

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

    dofs = lagrange_facet_fns(edge, deg - 1)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), dofs)
    xs = [immerse(cell, int_ned1, TrHCurl)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    center_dofs = lagrange_facet_fns(cell, deg - 2, interior=True, vector=True)

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

    ned = ElementTriple(cell, (nd, CellHCurl, C0), tri_dofs + center_dofs)
    return ned


def construct_tet_ndN(deg):
    cell = make_tetrahedron()
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]

    edge_dofs = lagrange_facet_fns(edge, deg - 1)
    int_ned = ElementTriple(edge, (PolynomialSpace(deg - 1, set_shape=True), CellHCurl, C0), edge_dofs)
    xs = [immerse(cell, int_ned, TrHCurl)]
    edge_dofs = [DOFGenerator(xs, tet_edges, S1)]

    face_dofs = lagrange_facet_fns(face, deg - 2, vector=True)
    face_ned = ElementTriple(face, (PolynomialSpace(deg - 2, set_shape=True), CellHCurl, C0), face_dofs)
    xs = [immerse(cell, face_ned, TrH1)]
    face_dofs = [DOFGenerator(xs, tet_faces, S1)]

    center_dofs = lagrange_facet_fns(cell, deg - 3, interior=True, vector=True)

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M1 = sp.Matrix([[0, z, -y]])
    M2 = sp.Matrix([[z, 0, -x]])
    M3 = sp.Matrix([[y, -x, 0]])

    vec_Pd = PolynomialSpace(deg - 1, set_shape=True)
    Pd = PolynomialSpace(deg - 1)
    nd_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M1 + (Pd.restrict(deg - 2, deg - 1))*M2 + (Pd.restrict(deg - 2, deg - 1))*M3

    ned = ElementTriple(cell, (nd_space, CellHCurl, C0), edge_dofs + face_dofs + center_dofs)
    assert len(ned.generate()) == (1/2)*deg*(deg + 2)*(deg + 3)
    return ned


def construct_tri_ndN_2(deg):
    cell = polygon(3)
    edge = cell.edges()[0]
    verts = cell.vertices(return_coords=True)
    verts.reverse()
    verts = [verts[0], verts[2], verts[1]]

    dofs = lagrange_facet_fns(edge, deg)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), dofs)
    xs = [immerse(cell, int_ned1, TrHCurl)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    dofs = vector_basis_fns(cell, deg - 1, rot=True)

    vec_Pk = PolynomialSpace(deg, set_shape=True)

    ned = ElementTriple(cell, (vec_Pk, CellHCurl, C0), tri_dofs + dofs)
    assert len(ned.generate()) == (deg + 1)*(deg + 2)
    return ned


def construct_tet_ndN_2(deg):
    cell = make_tetrahedron()
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]
    dofs = lagrange_facet_fns(edge, deg)
    edge_elem = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), dofs)
    xs = [immerse(cell, edge_elem, TrHCurl)]
    edge_dofs = [DOFGenerator(xs, tet_edges, S1)]

    face_dofs = []
    if deg >= 2:
        face_dofs = vector_basis_fns(face, deg - 1, rot=True)
        # not correct poly space
        face_elem = ElementTriple(face, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), face_dofs)
        xs = [immerse(cell, face_elem, TrH1)]
        face_dofs = [DOFGenerator(xs, tet_faces, S1)]

    center_dofs = []
    if deg >= 3:
        center_dofs = vector_basis_fns(cell, deg - 2, rot=True)

    vec_Pd = PolynomialSpace(deg, set_shape=True)

    nd2 = ElementTriple(cell, (vec_Pd, CellHCurl, C0), edge_dofs + face_dofs + center_dofs)
    assert len(nd2.generate()) == (1/2)*(deg + 1)*(deg + 2)*(deg + 3)
    return nd2

def construct_tri_rtN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    x = sp.Symbol("x")
    y = sp.Symbol("y")

    dofs = lagrange_facet_fns(edge, deg - 1)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHDiv, C0), dofs)
    xs = [immerse(cell, int_ned1, TrHDiv)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    center_dofs = lagrange_facet_fns(cell, deg - 2, interior=True, vector=True)

    x = sp.Symbol("x")
    y = sp.Symbol("y")

    M = sp.Matrix([[x, y]])
    vec_Pd = PolynomialSpace(deg - 1, set_shape=True)
    Pd = PolynomialSpace(deg - 1)
    rt = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    rt = ElementTriple(cell, (rt, CellHDiv, C0), tri_dofs + center_dofs)
    return rt


def construct_tet_rtN(deg):
    cell = make_tetrahedron()
    face = cell.d_entities(2)[0]

    face_dofs = lagrange_facet_fns(face, deg - 1)
    face_rt = ElementTriple(face, (PolynomialSpace(deg - 1, set_shape=True), CellHCurl, C0), face_dofs)
    xs = [immerse(cell, face_rt, TrHDiv)]
    face_dofs = [DOFGenerator(xs, tet_faces, S1)]

    center_dofs = lagrange_facet_fns(cell, deg - 2, interior=True, vector=True)

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M = sp.Matrix([[x, y, z]])

    vec_Pd = PolynomialSpace(deg - 1, set_shape=True)
    Pd = PolynomialSpace(deg - 1)
    rt_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    rt = ElementTriple(cell, (rt_space, CellHDiv, C0), face_dofs + center_dofs)
    assert len(rt.generate()) == (1/2)*deg*(deg + 1)*(deg + 3)
    return rt


def construct_tri_bdmN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    dofs = lagrange_facet_fns(edge, deg)
    edge_trip = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHDiv, C0), dofs)
    xs = [immerse(cell, edge_trip, TrHDiv)]
    tri_dofs = [DOFGenerator(xs, C3, S1)]

    center_dofs = vector_basis_fns(cell, deg - 1)

    vec_Pd = PolynomialSpace(deg, set_shape=True)

    bdm = ElementTriple(cell, (vec_Pd, CellHDiv, C0), tri_dofs + center_dofs)
    assert len(bdm.generate()) == (deg + 1)*(deg + 2)
    return bdm


def construct_tet_bdmN(deg):
    cell = make_tetrahedron()
    face = cell.d_entities(2)[0]

    dofs = lagrange_facet_fns(face, deg)
    int = ElementTriple(face, (PolynomialSpace(1, set_shape=True), CellHDiv, C0), dofs)
    xs = [immerse(cell, int, TrHDiv)]
    face_dofs = [DOFGenerator(xs, tet_faces, S1)]

    center_dofs = []
    if deg >= 2:
        center_dofs = vector_basis_fns(cell, deg - 1)

    vec_Pd = PolynomialSpace(deg, set_shape=True)

    bdm = ElementTriple(cell, (vec_Pd, CellHDiv, C0), face_dofs + center_dofs)
    assert len(bdm.generate()) == (1/2)*(deg + 1)*(deg + 2)*(deg + 3)
    return bdm


def construct_dgN(dim):
    def construct_dim_dgN(deg):
        return construct_dgNminus(dim)(deg + 1)
    return construct_dim_dgN


def construct_dgNminus(dim):
    if dim == 2:
        cell = polygon(3)
        inc = 3
    elif dim == 3:
        cell = make_tetrahedron()
        inc = 4
    else:
        raise NotImplementedError(f"Cell of dimension {dim} not implemented for DG")

    def construct_dim_dgNminus(deg):
        Pk = PolynomialSpace(deg)
        int_dofs = lagrange_facet_pts(cell, deg + inc)
        dgN = ElementTriple(cell, (Pk, CellL2, C0), int_dofs)
        assert len(dgN.generate()) == reduce(mul, [(deg + i)/i for i in range(1, dim + 1)])
        return dgN
    return construct_dim_dgNminus


# column: dimension: form number
constructors = {
    0: {
        2: {
            0: construct_tri_cgN,
            1: construct_tri_ndN,
            2: construct_tri_rtN,
            3: construct_dgNminus(2),
        },
        3: {
            0: construct_tet_cgN,
            1: construct_tet_ndN,
            2: construct_tet_rtN,
            3: construct_dgNminus(3),
        },
    },
    1: {
        2: {
            0: construct_tri_cgN,
            1: construct_tri_ndN_2,
            2: construct_tri_bdmN,
            3: construct_dgN(2),
        },
        3: {
            0: construct_tet_cgN,
            1: construct_tet_ndN_2,
            2: construct_tet_bdmN,
            3: construct_dgN(3),
        },
    },
}


def periodic_table(col, dim, k, deg):
    return constructors[col][dim][k](deg)
