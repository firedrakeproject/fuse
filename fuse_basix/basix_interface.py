from functools import cache
from fuse import Point, S1, S2
from sympy.combinatorics import Permutation
import numpy as np
import basix
from basix import MapType, SobolevSpace, create_custom_element, PolysetType, CellType


basix_elements = {"interval": CellType.interval,
                  "triangle": CellType.triangle}

# Keyed on ElementTriple.pullback().
# FUSE doesn't yet support doubleContravariantPiola or doubleCovariantPiola.
basix_mapping = {"identity": MapType.identity,
                 "covariant Piola": MapType.covariantPiola,
                 "contravariant Piola": MapType.contravariantPiola,
                 "L2 Piola": MapType.L2Piola}

# Keyed on ElementTriple.sobolev_space().
# FUSE doesn't yet support H3, HDivDiv, HEin or HInf.
basix_sobolev = {"L2": SobolevSpace.L2,
                 "H1": SobolevSpace.H1,
                 "HDiv": SobolevSpace.HDiv,
                 "HCurl": SobolevSpace.HCurl,
                 "H2": SobolevSpace.H2}


def _lookup(table, key, what):
    try:
        return table[key]
    except KeyError:
        raise NotImplementedError(f"{what} {key!r} has no Basix equivalent")


def check_vertex_correspondence(triple, cell_type):
    """Check that FUSE vertex i is sent to Basix reference vertex i.

    :func:`transform_to_basix_cell` relies on this, and so does matching
    entities by their vertices."""
    mapped = transform_to_basix_cell(triple.cell, triple.cell.vertices(return_coords=True))
    if not np.allclose(np.array(mapped, dtype=np.float64), basix.geometry(cell_type)):
        raise ValueError(f"The vertices of {triple.cell} do not correspond to the Basix reference {cell_type.name}")


def basix_to_fuse_affine_map(triple, cell_type):
    """The affine map ``x_fuse = J @ x_basix + b`` from the Basix reference
    cell onto the FUSE cell, sending Basix vertex i to FUSE vertex i."""
    v_basix = basix.geometry(cell_type)
    v_fuse = np.array(triple.cell.vertices(return_coords=True), dtype=np.float64)
    A = np.c_[v_basix, np.ones(len(v_basix))]
    sol, *_ = np.linalg.lstsq(A, v_fuse, rcond=None)
    J, b = sol[:-1].T, sol[-1]
    if not np.allclose(A @ sol, v_fuse):
        raise ValueError(f"{triple.cell} is not an affine image of the Basix reference {cell_type.name}")
    return J, b


def reference_value_map(triple, J):
    """The matrix K sending a function's values on the FUSE cell to the values
    of the corresponding function on the Basix cell, ``v_basix = K v_fuse``,
    as fixed by the element's pullback."""
    value_shape = tuple(triple.get_value_shape())
    value_size = int(np.prod(value_shape, dtype=int))
    pullback = triple.pullback()
    if pullback == "identity":
        return np.eye(value_size)
    if pullback == "L2 Piola":
        return np.linalg.det(J) * np.eye(value_size)
    if len(value_shape) != 1 or value_shape[0] != J.shape[0]:
        raise NotImplementedError(f"Pullback {pullback!r} with value shape {value_shape}")
    if pullback == "covariant Piola":
        return J.T
    if pullback == "contravariant Piola":
        return np.linalg.det(J) * np.linalg.inv(J)
    raise NotImplementedError(f"Pullback {pullback!r} has no Basix equivalent")


def compute_wcoeffs(triple, cell_type, J, b, K):
    """The element's polynomial space, as coefficients in Basix's orthonormal
    polynomial set on the Basix reference cell.

    FUSE's polynomial set lives on the FUSE cell, so it is pulled back onto the
    Basix cell (``v_basix(x) = K v_fuse(J x + b)``) and projected."""
    degree = triple.spaces[0].maxdegree
    pts, wts = basix.make_quadrature(cell_type, 2 * degree)
    basis = basix.polynomials.tabulate_polynomial_set(cell_type, PolysetType.standard, degree, 0, pts)[0]

    sd = triple.ref_el.get_spatial_dimension()
    vals = triple.poly_set.tabulate(pts @ J.T + b, 0)[(0,) * sd]
    vals = vals.reshape(vals.shape[0], K.shape[1], -1)  # (member, value, point)
    vals = np.einsum("ab,nbq->naq", K, vals)

    wcoeffs = np.einsum("naq,jq,q->naj", vals, basis, wts).reshape(vals.shape[0], -1)
    # orthonormalise the rows; this keeps their span
    q, _ = np.linalg.qr(wcoeffs.T)
    return np.ascontiguousarray(q.T)


def basix_entity_alignment(triple, cell_type):
    """Match each Basix reference entity to a FUSE entity and orientation.

    Entities are matched by their vertex sets, since FUSE and Basix may
    number them differently. The orientation is the member ``o`` of the FUSE
    entity's group that reorders the entity's FUSE vertex order into Basix's
    order, so that ``o.permute(fuse_vertices) == basix_vertices``.

    :returns: ``alignment[dim][basix_entity] = (fuse_entity, fiat_entity, o)``
    """
    fuse_top = triple.ref_el.get_topology()
    basix_top = basix.topology(cell_type)
    min_ids = triple.cell.get_starter_ids()
    fuse_entities = {dim: {e.id - min_ids[dim]: e for e in triple.cell.d_entities(dim)}
                     for dim in fuse_top}

    alignment = []
    for dim, entities in enumerate(basix_top):
        aligned = []
        for t_basix in entities:
            matches = [e_F for e_F, t_fuse in fuse_top[dim].items() if set(t_fuse) == set(t_basix)]
            if len(matches) != 1:
                raise ValueError(f"No unique FUSE entity with vertices {t_basix}")
            e_F = matches[0]
            entity = fuse_entities[dim][e_F]
            t_fuse = fuse_top[dim][e_F]
            if dim == 0 or tuple(t_fuse) == tuple(t_basix):
                o = entity.group.identity
            else:
                o = entity.group.get_member(Permutation([t_fuse.index(v) for v in t_basix]))
            aligned.append((entity, e_F, o))
        alignment.append(aligned)
    return alignment


def compute_points_and_matrices(triple, cell_type, K=None):
    """Interpolation points and matrices in the layout Basix expects.

    Basix computes the DOF transformations of an element from the DOFs on
    one entity of each type, so it needs the DOFs on every entity to be the
    same functionals relative to that entity's reference orientation (its
    vertex order in the Basix topology). FUSE defines each entity's DOFs
    relative to its own vertex order, so the DOFs of an entity whose FUSE
    order differs are rewritten in the Basix orientation, using FUSE's
    action of the reorienting symmetry on them.

    As a result, Basix's DOF numbering can differ from FUSE/FIAT numbering
    within reoriented entities.

    The DOFs are also moved from the FUSE cell onto the Basix cell: points
    by the affine map, and weights by the element's pullback, so that each
    DOF takes the same value on corresponding functions.

    :param: K: value map from :func:`reference_value_map`; computed if not given.
    :returns: ``x`` indexed ``(tdim, entity, point, dim)`` and ``M`` indexed
        ``(tdim, entity, dof, value, point, derivative)``.
    """
    triple.to_ufl()  # ensures ref_el, entity_ids and nodes are set up
    check_vertex_correspondence(triple, cell_type)
    if K is None:
        J, _ = basix_to_fuse_affine_map(triple, cell_type)
        K = reference_value_map(triple, J)
    # A DOF sum_p w_p . v(x_p) equals sum_p (K^-T w_p) . v_basix(x_p)
    weight_map = np.linalg.inv(K).T
    spdim = triple.ref_el.get_spatial_dimension()
    value_shape = tuple(triple.get_value_shape())
    value_size = int(np.prod(value_shape, dtype=int))

    xs = [[] for _ in range(4)]
    Ms = [[] for _ in range(4)]
    for dim, aligned in enumerate(basix_entity_alignment(triple, cell_type)):
        for entity, e_F, o in aligned:
            nodes = [triple.nodes[i] for i in triple.entity_ids[dim][e_F]]
            if any(node.max_deriv_order > 0 for node in nodes):
                raise NotImplementedError("Derivative DOFs are not yet supported in the Basix conversion")

            # Points for every DOF on this entity, each DOF reading only its own.
            pts = [pt for node in nodes for pt in node.pt_dict]
            M = np.zeros((len(nodes), value_size, len(pts), 1))
            p = 0
            for k, node in enumerate(nodes):
                for pt in node.pt_dict:
                    for weight, comp in node.pt_dict[pt]:
                        c = np.ravel_multi_index(comp, value_shape) if value_shape else 0
                        M[k, c, p, 0] += weight
                    p += 1
            M = np.einsum("ab,kbpd->kapd", weight_map, M)

            if nodes and not o.perm.is_Identity:
                # DOF i in the Basix orientation, as a combination of the FUSE DOFs
                _, T = triple.entity_dof_action(entity, o)
                M = np.einsum("ij,j...->i...", T, M)

            if pts:
                x = np.array(transform_to_basix_cell(triple.cell, pts), dtype=np.float64)
            else:
                x = np.zeros((0, spdim))
            xs[dim].append(np.ascontiguousarray(x))
            Ms[dim].append(np.ascontiguousarray(M))
    return xs, Ms


def convert_to_basix_element(triple):
    """Convert a FUSE element to a Basix custom element.

    The DOFs on each entity are expressed relative to the Basix reference
    orientation of that entity, see :func:`compute_points_and_matrices`."""
    ref_el = triple.cell.to_fiat()
    cell_type = _lookup(basix_elements, ref_el.cellname(), "Cell")
    J, b = basix_to_fuse_affine_map(triple, cell_type)
    K = reference_value_map(triple, J)
    x, M = compute_points_and_matrices(triple, cell_type, K)
    value_shape = list(triple.get_value_shape())
    wcoeffs = compute_wcoeffs(triple, cell_type, J, b, K)
    return create_custom_element(cell_type,
                                 value_shape,
                                 wcoeffs,
                                 x,
                                 M,
                                 0,  # skip derivative for now
                                 _lookup(basix_mapping, triple.pullback(), "Pullback"),
                                 _lookup(basix_sobolev, triple.sobolev_space(), "Sobolev space"),
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


def ufc_interval():
    return Point(1, [Point(0), Point(0)], vertex_num=2, variant="ufc", group=S2)


basix_cell_constructors = {"interval": ufc_interval,
                           "triangle": ufc_triangle}


@cache
def basix_fuse_cell(cellname):
    """FUSE representation of the Basix reference cell with this name.

    Built on first use rather than at import, so that importing this
    module does not depend on cell construction succeeding."""
    try:
        constructor = basix_cell_constructors[cellname]
    except KeyError:
        raise NotImplementedError(f"Fuse cell {cellname} doesn't have a Basix equivalent")
    return constructor()


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
    basix_cell = basix_fuse_cell(fuse_cell.to_ufl().cellname)
    return transform_points(fuse_cell, basix_cell, x)
