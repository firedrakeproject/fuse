import re

from fuse import *

tri_cells = [
    [(5, 0), (4.40032272, -0.60452232), (4, 0)],
    [(5, 0), (5.00036392, -0.8645401), (4.40032272, -0.60452232)],
    [(4.40032272, -0.60452232), (4.40889435, -1.20170631), (5.00036392, -0.8645401)],
    [(5.00036392, -0.8645401), (4.94321671, -1.53887215), (4.40889435, -1.20170631)],
    [(4.40889435, -1.20170631), (3.74599228, -0.94740295), (4.40032272, -0.60452232)],
    [(3.63169861, -0.46215935), (3.74599228, -0.94740295), (4.40032272, -0.60452232)],
    [(4, 0), (3.63169861, -0.46215935), (4.40032272, -0.60452232)],
    [(4, 0), (3, 0), (3.63169861, -0.46215935)],
    [(3.63169861, -0.46215935), (3.01165619, -0.74503593), (3.74599228, -0.94740295)],
    [(3.74599228, -0.94740295), (3.13452148, -1.33079071), (3.01165619, -0.74503593)],
    [(3, 0), (3.01165619, -0.74503593), (3.63169861, -0.46215935)],
    [(3.01165619, -0.74503593), (2.428759, -0.47073174), (3, 0)],
    [(3.01165619, -0.74503593), (2.30303574, -0.96505127), (2.428759, -0.47073174)],
    [(3, 0), (2, 0), (2.428759, -0.47073174)],
    [(2.428759, -0.47073174), (1.74585495, -0.55930901), (2, 0)],
    [(2, 0), (1, 0), (1.74585495, -0.55930901)],
    [(1.74585495, -0.55930901), (0.97151604, -0.55359421), (1, 0)],
    [(0.97151604, -0.55359421), (0, 0), (1, 0)],
    [(1.74585495, -0.55930901), (2.30303574, -0.96505127), (2.428759, -0.47073174)],
]

quad_cells = [
    [(0, 0), (0.48999329, -0.0172266), (0.7619957, -0.63071404), (0.20733509, -0.4371994)],
    [(0.48999329, -0.0172266), (1, 0), (1.28719101, -0.38296623), (0.7619957, -0.63071404)],
    [(1.28719101, -0.38296623), (1, 0), (2, 0), (1.78053589, -0.54589462)],
    [(1.78053589, -0.54589462), (2.37284546, -0.37569046), (3, 0), (2, 0)],
    [(2.37284546, -0.37569046), (2.94717102, -0.48426933), (2.52142715, -0.92429962), (1.78053589, -0.54589462)],
    [(3, 0), (4, 0), (2.94717102, -0.48426933), (2.37284546, -0.37569046)],
    [(2.94717102, -0.48426933), (3.703862, -0.5842762), (3.11575394, -1.02430649), (2.52142715, -0.92429962)],
    [(3.703862, -0.5842762), (4.26197433, -1.03829689), (3.43291855, -1.25003624), (3.11575394, -1.02430649)],
    [(3.43291855, -1.25003624), (3.82432404, -1.65318604), (4.35512314, -1.76211014), (4.26197433, -1.03829689)],
    [(4.26197433, -1.03829689), (4.98679276, -0.76590099), (5.02659836, -1.862117), (4.35512314, -1.76211014)],
    [(4, 0), (4.02772179, -0.36404972), (3.703862, -0.5842762), (2.94717102, -0.48426933)],
    [(3.703862, -0.5842762), (4.02772179, -0.36404972), (4.98679276, -0.76590099), (4.26197433, -1.03829689)],
    [(4, 0), (5, 0), (4.98679276, -0.76590099), (4.02772179, -0.36404972)],
]

elem = periodic_table(1, 2, 1, 2)

def construct_hermite():
    tri = polygon(3)
    vert = tri.vertices()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    v_xs = [immerse(tri, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, S3/S2, S1)

    v_derv_xs = [immerse(tri, dg0, TrGrad)]
    v_derv_dofs = DOFGenerator(v_derv_xs, S3/S2, S1)

    v_derv2_xs = [immerse(tri, dg0, TrHess)]
    v_derv2_dofs = DOFGenerator(v_derv2_xs, S3/S2, S1)

    i_xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    i_dofs = DOFGenerator(i_xs, S1, S1)

    her = ElementTriple(tri, (P3, CellH2, C0),
                        [v_dofs, v_derv_dofs, v_derv2_dofs, i_dofs])
    return her
# elem = construct_hermite()

total_tikz = []
for cell in tri_cells:
    total_tikz += elem.to_tikz(show=False, vertices=cell)


_COORD_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)"
)


def _coords(cmd):
    """Numeric (x, y) tuples in a tikz command (unit-bearing ones like (2pt) are skipped)."""
    return [(float(x), float(y)) for x, y in _COORD_RE.findall(cmd)]


def _is_edge(cmd):
    return cmd.startswith("\\draw[thick] ")


def _pt_key(p, ndig=6):
    return (round(p[0], ndig), round(p[1], ndig))


def _edge_key(a, b, ndig=6):
    return tuple(sorted((_pt_key(a, ndig), _pt_key(b, ndig))))


def _on_segment(p, a, b, tol=1e-6):
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    length_sq = abx * abx + aby * aby
    if length_sq == 0:
        return False
    if abs(abx * apy - aby * apx) > tol * length_sq ** 0.5:
        return False
    t = (apx * abx + apy * aby) / length_sq
    return tol < t < 1 - tol


def remove_repeated_edges(tikz_commands):
    """Drop edges and vertices shared by more than one cell, keeping the first
    drawn instance, and remove the DOFs sitting on those repeated entities.

    Each cell contributes its edge commands followed by its DOF commands. Edges
    are matched by their (order-independent) endpoints; an edge DOF is tied to an
    edge when its point lies on that edge segment, and a vertex DOF when its point
    coincides with an edge endpoint. The full set of DOFs at a shared vertex is
    kept from the first cell that draws it.
    """
    result = []
    seen_edges = set()
    seen_vertices = set()

    def process(cell_edges, cell_dofs):
        duplicate_edges = []
        cell_vertices = set()
        for cmd in cell_edges:
            a, b = _coords(cmd)[:2]
            cell_vertices.update((_pt_key(a), _pt_key(b)))
            key = _edge_key(a, b)
            if key in seen_edges:
                duplicate_edges.append((a, b))
            else:
                seen_edges.add(key)
                result.append(cmd)
        new_vertices = set()
        for cmd in cell_dofs:
            pts = _coords(cmd)
            if not pts:
                result.append(cmd)
                continue
            if any(_on_segment(pts[0], a, b) for a, b in duplicate_edges):
                continue
            vkey = _pt_key(pts[0])
            if vkey in cell_vertices:
                if vkey in seen_vertices:
                    continue
                new_vertices.add(vkey)
            result.append(cmd)
        seen_vertices.update(new_vertices)

    cell_edges, cell_dofs = [], []
    saw_dof = False
    for cmd in tikz_commands:
        if _is_edge(cmd):
            if saw_dof:
                process(cell_edges, cell_dofs)
                cell_edges, cell_dofs = [], []
                saw_dof = False
            cell_edges.append(cmd)
        else:
            cell_dofs.append(cmd)
            saw_dof = True
    process(cell_edges, cell_dofs)
    return result


total_tikz = remove_repeated_edges(total_tikz)
print("\n".join(total_tikz))