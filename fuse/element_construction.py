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


def points_close(p, q, tol=1e-6):
    return np.linalg.norm(p - q) < tol


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


def group_with_mappings(points, verts, return_idx=False, tol=1e-6):
    points = [np.array(p) for p in points]

    # compute barycentric coordinates
    bary = [barycentric_coords(p, verts) for p in points]

    # first group using fast method
    raw_groups = group_by_symmetry(points, verts, tol)
    result = {}

    for group in raw_groups:
        if len(bary[0]) > 2:
            if len(group) < len(list(itertools.permutations(range(len(verts))))):
                # need to ensure the base is chosen to be invariant under S2
                # S2 reflects the 0th and 1st barycentric components (on a triangle)
                # Pick generators that lie on axis of symmetry
                base = [i for i in group if np.allclose(bary[i][0], bary[i][1])][0]
                # base = [i for i in group if np.allclose(bary[i][1], bary[i][2])][0]
            else:
                # if the group generates all permutations this may not matter... or at least the choice is different TODO
                # choose the generators nearest to the fixed vertex
                base = sorted(group, key=lambda i: bary[i][2])[-1]
                # base = sorted(group, key=lambda i: bary[i][0])[-1]
        else:
            base = group[0]
        lam_base = bary[base]

        perm_list = []
        for j in group:
            lam_j = bary[j]

            perm = find_permutation(lam_base, lam_j, tol)

            if perm is None:
                raise ValueError("Failed to find permutation")

            perm_list += [Permutation(perm)]

        perm_group = sp.combinatorics.PermutationGroup(perm_list)
        if perm_group.order() != len(perm_list):
            perm_group = PermutationSetRepresentation(perm_list)
        else:
            perm_group = GroupRepresentation(perm_group)

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


def barycentric_gradients(cell):
    """The barycentric coordinates of ``cell`` as symbols ``s_i``, one for each
    vertex in ``cell.ordered_vertices()``, and their gradients."""
    symbols = list(sp.symbols(f"s_0:{cell.dimension + 1}"))
    coords = symbols[:-1]
    v_0 = np.array(cell.ordered_vertex_coords()[0])
    bvs = np.array(cell.basis_vectors(norm=False))
    res = np.matmul(np.linalg.inv(bvs.T), np.array(coords - v_0))
    ls = (1 - sum(res),) + tuple(res[i] for i in range(len(res)))
    dl = [sp.Matrix([sp.diff(l, x) for x in coords]) for l in ls]
    return symbols, dl


def whitney_bubbles(cell, entity):
    """The Whitney fields of the edges of ``entity`` from its first vertex, each
    times the barycentric coordinates of the other vertices of ``entity``, in
    the barycentric coordinates of ``cell``.

    With the vertices of ``entity`` at positions v_0, ..., v_n in
    ``cell.ordered_vertices()``, and the Whitney field of the edge (i, j)

        W_ij = l_i grad(l_j) - l_j grad(l_i),

    these are, for t = 1, ..., n,

        B_t = (prod over s = 1, ..., n, s != t of l_{v_s}) W_{v_0 v_t}.

    On a triangle (0, 1, 2) this gives l_2 W_01 and l_1 W_02; on a tetrahedron
    l_2 l_3 W_01, l_1 l_3 W_02 and l_1 l_2 W_03; on an edge just W_01.

    They vanish tangentially on the boundary of ``entity``, and span the same
    space as the fields made in this way from any of its vertices, so every
    symmetry of ``entity`` preserves their span.
    """
    s, dl = barycentric_gradients(cell)
    v = [cell.ordered_vertices().index(u) for u in entity.ordered_vertices()]
    return [sp.Matrix(math.prod(s[w] for w in v[1:] if w != t) * (s[v[0]]*dl[t] - s[t]*dl[v[0]]))
            for t in v[1:]]


def nedelec_bubble_fns(cell, entity, deg, transversal=None, action=None):
    """Moments against the Nedelec (first kind) functions of degree ``deg`` that
    vanish tangentially on the boundary of ``entity``: its Whitney bubbles times
    the polynomials of degree ``deg - entity.dimension`` on it.

    For a sub-entity, the polynomial in each orientation is listed and
    ``transversal`` generates the moments on the other entities like it.
    """
    symbols, _ = barycentric_gradients(cell)
    fields = whitney_bubbles(cell, entity)
    pos = [cell.ordered_vertices().index(v) for v in entity.ordered_vertices()]
    basis_funcs, groups, lg_symbols = lagrange_barycentric_basis(entity.dimension, entity.ordered_vertex_coords(),
                                                                 deg - entity.dimension)
    dofs = []
    for bf, grp in zip(basis_funcs, groups):
        grp = grp.add_cell(entity)
        on_cell = lambda o: bf.as_expr().xreplace({o_sym: symbols[p] for o_sym, p in zip(o.permute(lg_symbols), pos)})
        orientations = [grp.identity] if entity == cell else grp.members()
        xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(f*on_cell(o), symbols=symbols, action=action))
              for o in orientations for f in fields]
        dofs += [DOFGenerator(xs, grp if entity == cell else transversal)]
    return dofs


def proxy_field_bfs(cell, rot=False):
    symbols, dl = barycentric_gradients(cell)
    edges = [(0, 1), (1, 2), (0, 2)]
    bfs = [sp.Matrix(symbols[i]*dl[j] - symbols[j]*dl[i]) for (i, j) in edges]
    facet_syms = [[symbols[i] for i in facet] for facet in edges]
    if cell.dimension == 2:
        grp = [C3]
    else:
        bfs = [sp.Matrix(symbols[i]*dl[j] - symbols[j]*dl[i]) for (i, j) in [(1, 2), (2, 0), (0, 1), (3, 1), (2, 3), (0, 3)]]
        grp = [tet_edges]
        facet_syms = [[symbols[i] for i in facet] for facet in [(1, 2), (2, 0), (0, 1), (3, 1), (2, 3), (0, 3)]]
    if rot:
        if cell.dimension == 2:
            perp = lambda x: np.array([[0, -1], [1, 0]]) @ x
            bfs = [perp(bf) for bf in bfs]
            grp = [C3]
        else:
            bfs = [sp.Matrix(symbols[i]*dl[j].cross(dl[k]) + symbols[j]*dl[k].cross(dl[i]) + symbols[k]*dl[i].cross(dl[j])) for i, j, k in [[0, 3, 1], [3, 1, 2], [0, 1, 2], [0, 2, 3]]]
            grp = [tet_faces]
            facet_syms = [[symbols[i] for i in facet] for facet in [[0, 3, 1], [3, 1, 2], [0, 1, 2], [0, 2, 3]]]
    if cell.dimension == 3:
        # As for the immersed DOFs, the field of the first edge (face) is
        # generated onto the others by the transversal.
        base = cell.d_entities(2 if rot else 1)[0]
        base_ids = {cell.ordered_vertices().index(v) for v in base.ordered_vertices()}
        k = [i for i, fs in enumerate(facet_syms) if {symbols.index(x) for x in fs} == base_ids][0]
        bfs, facet_syms = [bfs[k]], [facet_syms[k]]

    return bfs, grp, symbols, facet_syms


def vector_basis_fns(cell, deg, rot=False, action=None):
    """
    Returns dofs that are moments against vector valued basis functions of a given degree over a given cell,
    by default returning Nedelec basis functions, and returning Raviart Thomas if rot=True

    ``action`` is the action of the kernels (see :mod:`fuse.actions`).
    """
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]
    facet_cell = edge
    if cell.dimension == 3 and rot:
        facet_cell = face
    dofs = []
    pf_basis_funcs, pf_grps, symbols, facet_syms = proxy_field_bfs(cell, rot)
    basis_funcs, groups, lg_symbols = lagrange_barycentric_basis(facet_cell.dimension, facet_cell.ordered_vertex_coords(), deg - 1)
    for pf_bf, pf_grp, facet_sym in zip(pf_basis_funcs, pf_grps, facet_syms):
        for bf, grp in zip(basis_funcs, groups):
            grp = grp.add_cell(facet_cell)
            subs_dict = lambda o: {o_sym: p_sym for o_sym, p_sym in zip(o.permute(lg_symbols), facet_sym)}
            new_bf = lambda o: bf.as_expr().xreplace(subs_dict(o))
            if grp.size() == 2 and cell.dimension == 2:
                iden = grp.identity
                xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel(pf_bf*new_bf(iden), symbols=symbols, action=action))]
                dofs += [DOFGenerator(xs1, grp*pf_grp)]
            else:
                # the orientations of one moment on its entity form one generator
                xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel(pf_bf*new_bf(o), symbols=symbols, action=action))
                       for o in grp.members()]
                dofs += [DOFGenerator(xs1, pf_grp)]

    if not rot:
        if cell.dimension == 3 and deg >= 2:
            dofs += nedelec_bubble_fns(cell, face, deg, tet_faces, action=action)
        if deg >= cell.dimension:
            dofs += nedelec_bubble_fns(cell, cell, deg, action=action)
        return dofs

    basis_funcs, groups, symbols = lagrange_barycentric_basis(cell.dimension, cell.ordered_vertex_coords(), deg - 2)
    v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
    v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
    v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))
    vector_basis = [(v_0 - v_2)/2, (v_0 - v_1)/2]
    if cell.dimension == 3:
        v_3 = np.array(cell.get_node(cell.ordered_vertices()[3], return_coords=True))
        vector_basis += [(v_0 - v_3)/2]
    vgrps = [S1 for _ in vector_basis]
    for bf, grp in zip(basis_funcs, groups):
        if cell.dimension == 3:
            # the three directions span the constant vectors, which every
            # symmetry preserves, so they belong together in one generator
            xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*vec_bf*bf, symbols=symbols, action=action))
                   for vec_bf in vector_basis]
            dofs += [DOFGenerator(xs1, grp.add_cell(cell))]
            continue
        tempvgrps = vgrps
        if grp.size()*2 <= cell.group.size():
            tempvgrps = [S2]
        for vec_bf, vgrp in zip(vector_basis, tempvgrps):
            vgrp = vgrp.add_cell(cell)
            grp = grp.add_cell(cell)
            if grp.size()*vgrp.size() <= cell.group.size():
                xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*vec_bf*bf, symbols=symbols, action=action))]
                if grp.size() == 1:
                    # the images under vgrp span an invariant space on their
                    # own, so they belong together in one generator
                    xs1 = [x(s) for x in xs1 for s in cell.group._match(vgrp)]
                    dofs += [DOFGenerator(xs1, grp)]
                else:
                    dofs += [DOFGenerator(xs1, grp*vgrp)]
            else:
                xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel(np.prod(symbols)*vec_bf*bf, symbols=symbols, action=action))]
                dofs += [DOFGenerator(xs1, grp)]
    return dofs


def lagrange_facet_fns(cell, deg, vector=False, action=None):
    dofs = []
    basis_funcs, groups, symbols = lagrange_barycentric_basis(cell.dimension, cell.ordered_vertex_coords(), deg)
    for bf, grp in zip(basis_funcs, groups):
        if not vector:
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf, symbols=symbols, action=action))]
            dofs += [DOFGenerator(xs, grp)]
        else:
            v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
            v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
            v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))
            xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_2)/2, symbols=symbols, action=action))]
            if cell.dimension == 3:
                # The three directions are moved into each other by the
                # symmetries, so they belong to one generator: on their own
                # they need not span an invariant space.
                v_3 = np.array(cell.get_node(cell.ordered_vertices()[3], return_coords=True))
                xs += [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_1)/2, symbols=symbols, action=action)),
                       DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_3)/2, symbols=symbols, action=action))]
                dofs += [DOFGenerator(xs, grp)]
            else:
                if grp.size() > 3:
                    dofs += [DOFGenerator(xs, grp)]
                    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(bf*(v_0 - v_1)/2, symbols=symbols, action=action))]
                    dofs += [DOFGenerator(xs, grp)]
                elif grp.size() == 1:
                    # the two directions span the constant vectors, an
                    # invariant space, so they belong together in one generator
                    xs = [x(s) for x in xs for s in cell.group._match(S2.add_cell(cell))]
                    dofs += [DOFGenerator(xs, grp)]
                else:
                    dofs += [DOFGenerator(xs, S2*grp)]

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
            int_dofs += [DOFGenerator([d], grp) for d in dofs]
    return int_dofs


def rt_space(dim, deg):
    """The Raviart-Thomas space of degree ``deg`` (lowest order 1) on a
    ``dim``-dimensional simplex."""
    x = sp.symbols("x y z")[:dim]
    Pd = PolynomialSpace(deg - 1)
    return PolynomialSpace(deg - 1, shape=dim) + Pd.restrict(deg - 2, deg - 1)*sp.Matrix([x])


def construct_tri_cgN(deg):
    cell = polygon(3)
    vert = cell.vertices()[0]
    edge = cell.edges()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, C0, Fid), DOFGenerator(xs, S1))
    v_xs = [immerse(cell, dg0)]
    v_dofs = [DOFGenerator(v_xs, C3)]

    points = recursive_nodes(1, deg, domain="equilateral")[1:-1].flatten()

    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    sym_dofs = [DOFGenerator([pt], S2) for pt in sym_points]
    if 0 in points:
        centre_dof = [DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1)]
    else:
        centre_dof = []
    edge_dofs = []
    if deg >= 2:
        edge_dg0 = ElementTriple(edge, (PolynomialSpace(deg - 2), C0, Fid), sym_dofs + centre_dof)
        edge_dofs = [DOFGenerator([immerse(cell, edge_dg0)], C3)]

    int_dofs = lagrange_facet_pts(cell, deg)
    return ElementTriple(cell, (Pk, C0, Fid), v_dofs + edge_dofs + int_dofs)


def construct_tet_cgN(deg):
    cell = make_tetrahedron()
    vert = cell.vertices()[0]
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, C0, Fid), DOFGenerator(xs, S1))
    v_xs = [immerse(cell, dg0)]
    v_dofs = [DOFGenerator(v_xs, C4)]

    points = recursive_nodes(1, deg, domain="equilateral")[1:-1].flatten()

    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    sym_dofs = [DOFGenerator([pt], S2) for pt in sym_points]
    if 0 in points:
        centre_dof = [DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1)]
    else:
        centre_dof = []
    edge_dofs = []
    if deg >= 2:
        edge_dg0 = ElementTriple(edge, (PolynomialSpace(deg - 2), C0, Fid), sym_dofs + centre_dof)
        edge_dofs = [DOFGenerator([immerse(cell, edge_dg0)], tet_edges)]

    face_dofs = []
    if deg >= 3:
        face_trp = ElementTriple(face, (PolynomialSpace(deg - 3), C0, Fid), lagrange_facet_pts(face, deg))
        face_dofs = [DOFGenerator([immerse(cell, face_trp)], tet_faces)]

    int_dofs = lagrange_facet_pts(cell, deg)

    cg = ElementTriple(cell, (Pk, C0, Fid), v_dofs + edge_dofs + face_dofs + int_dofs)
    assert len(cg.generate()) == (deg + 1)*(deg + 2)*(deg + 3)/6
    return cg


def construct_tri_ndN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    dofs = lagrange_facet_fns(edge, deg - 1, action=HCurl)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(deg - 1), L2, Fcurl), dofs)
    xs = [immerse(cell, int_ned1)]
    tri_dofs = [DOFGenerator(xs, C3)]

    center_dofs = lagrange_facet_fns(cell, deg - 2, vector=True, action=HCurl)

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    vec_Pk = PolynomialSpace(deg - 1, shape=2)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

    ned = ElementTriple(cell, (nd, C0, Fcurl), tri_dofs + center_dofs)
    return ned


def construct_tet_ndN(deg):
    cell = make_tetrahedron()
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]

    edge_dofs = lagrange_facet_fns(edge, deg - 1, action=HCurl)
    int_ned = ElementTriple(edge, (PolynomialSpace(deg - 1), L2, Fcurl), edge_dofs)
    xs = [immerse(cell, int_ned)]
    edge_dofs = [DOFGenerator(xs, tet_edges)]

    face_dofs = []
    if deg >= 2:
        face_ned = ElementTriple(face, (PolynomialSpace(deg - 2, shape=2), L2, Fcurl),
                                 lagrange_facet_fns(face, deg - 2, vector=True, action=HCurl))
        face_dofs = [DOFGenerator([immerse(cell, face_ned)], tet_faces)]

    center_dofs = lagrange_facet_fns(cell, deg - 3, vector=True, action=HCurl)

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M1 = sp.Matrix([[0, z, -y]])
    M2 = sp.Matrix([[z, 0, -x]])
    M3 = sp.Matrix([[y, -x, 0]])

    vec_Pd = PolynomialSpace(deg - 1, shape=3)
    Pd = PolynomialSpace(deg - 1)
    nd_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M1 + (Pd.restrict(deg - 2, deg - 1))*M2 + (Pd.restrict(deg - 2, deg - 1))*M3

    ned = ElementTriple(cell, (nd_space, C0, Fcurl), edge_dofs + face_dofs + center_dofs)
    assert len(ned.generate()) == (1/2)*deg*(deg + 2)*(deg + 3)
    return ned


def construct_tri_ndN_2(deg):
    cell = polygon(3)
    edge = cell.edges()[0]
    verts = cell.vertices(return_coords=True)
    verts.reverse()
    verts = [verts[0], verts[2], verts[1]]

    dofs = lagrange_facet_fns(edge, deg, action=HCurl)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(deg), L2, Fcurl), dofs)
    xs = [immerse(cell, int_ned1)]
    tri_dofs = [DOFGenerator(xs, C3)]

    dofs = vector_basis_fns(cell, deg - 1, rot=True, action=HCurl)

    vec_Pk = PolynomialSpace(deg, shape=2)

    ned = ElementTriple(cell, (vec_Pk, C0, Fcurl), tri_dofs + dofs)
    assert len(ned.generate()) == (deg + 1)*(deg + 2)
    return ned


def construct_tet_ndN_2(deg):
    cell = make_tetrahedron()
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]
    dofs = lagrange_facet_fns(edge, deg, action=HCurl)
    edge_elem = ElementTriple(edge, (PolynomialSpace(deg), L2, Fcurl), dofs)
    xs = [immerse(cell, edge_elem)]
    edge_dofs = [DOFGenerator(xs, tet_edges)]

    face_dofs = []
    if deg >= 2:
        face_dofs = vector_basis_fns(face, deg - 1, rot=True, action=HCurl)
        face_elem = ElementTriple(face, (rt_space(2, deg - 1), L2, Fcurl), face_dofs)
        xs = [immerse(cell, face_elem)]
        face_dofs = [DOFGenerator(xs, tet_faces)]

    center_dofs = []
    if deg >= 3:
        center_dofs = vector_basis_fns(cell, deg - 2, rot=True, action=HCurl)

    vec_Pd = PolynomialSpace(deg, shape=3)

    nd2 = ElementTriple(cell, (vec_Pd, C0, Fcurl), edge_dofs + face_dofs + center_dofs)
    assert len(nd2.generate()) == (1/2)*(deg + 1)*(deg + 2)*(deg + 3)
    return nd2


def construct_tri_rtN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    dofs = lagrange_facet_fns(edge, deg - 1, action=HDiv)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(deg - 1), L2, Fdiv), dofs)
    xs = [immerse(cell, int_ned1)]
    tri_dofs = [DOFGenerator(xs, C3)]

    center_dofs = lagrange_facet_fns(cell, deg - 2, vector=True, action=HDiv)

    rt = ElementTriple(cell, (rt_space(2, deg), C0, Fdiv), tri_dofs + center_dofs)
    return rt


def construct_tet_rtN(deg):
    cell = make_tetrahedron()
    face = cell.d_entities(2)[0]

    face_dofs = lagrange_facet_fns(face, deg - 1, action=HDiv)
    face_rt = ElementTriple(face, (PolynomialSpace(deg - 1), L2, Fdiv), face_dofs)
    xs = [immerse(cell, face_rt)]
    face_dofs = [DOFGenerator(xs, tet_faces)]

    center_dofs = lagrange_facet_fns(cell, deg - 2, vector=True, action=HDiv)

    rt = ElementTriple(cell, (rt_space(3, deg), C0, Fdiv), face_dofs + center_dofs)
    assert len(rt.generate()) == (1/2)*deg*(deg + 1)*(deg + 3)
    return rt


def construct_tri_bdmN(deg):
    cell = polygon(3)
    edge = cell.edges()[0]

    dofs = lagrange_facet_fns(edge, deg, action=HDiv)
    edge_trip = ElementTriple(edge, (PolynomialSpace(deg), L2, Fdiv), dofs)
    xs = [immerse(cell, edge_trip)]
    tri_dofs = [DOFGenerator(xs, C3)]

    center_dofs = vector_basis_fns(cell, deg - 1, action=HDiv)

    vec_Pd = PolynomialSpace(deg, shape=2)

    bdm = ElementTriple(cell, (vec_Pd, C0, Fdiv), tri_dofs + center_dofs)
    assert len(bdm.generate()) == (deg + 1)*(deg + 2)
    return bdm


def construct_tet_bdmN(deg):
    cell = make_tetrahedron()
    face = cell.d_entities(2)[0]

    dofs = lagrange_facet_fns(face, deg, action=HDiv)
    face_bdm = ElementTriple(face, (PolynomialSpace(deg), L2, Fdiv), dofs)
    xs = [immerse(cell, face_bdm)]
    face_dofs = [DOFGenerator(xs, tet_faces)]

    center_dofs = []
    if deg >= 2:
        center_dofs = vector_basis_fns(cell, deg - 1, action=HDiv)

    vec_Pd = PolynomialSpace(deg, shape=3)

    bdm = ElementTriple(cell, (vec_Pd, C0, Fdiv), face_dofs + center_dofs)
    assert len(bdm.generate()) == (1/2)*(deg + 1)*(deg + 2)*(deg + 3)
    bdm.to_fiat()
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
        dgN = ElementTriple(cell, (Pk, C0, Fid), int_dofs)
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
