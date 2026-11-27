#!/usr/bin/env python3
"""
Generate a 7-cell hexahedral ring with a 180-degree twist.

The combinatorics implement the first 3-D counterexample in:
Agelek, Anderson, Bangerth, Barth,
"On orienting edges of unstructured two- and three-dimensional meshes".

The mesh is written in Gmsh 2.2 ASCII format and includes a parity-based
checker proving that the usual "opposite edges have parallel directions"
constraint is unsatisfiable.

Usage:
    python generate_nonorientable_hex_ring.py [output.msh]
"""

import math
import sys
from collections import defaultdict, deque
from pathlib import Path

N = 7
R = 3.0
A = 0.6

# Gmsh hex vertex convention:
# bottom: 0,1,2,3 cyclic; top: 4,5,6,7 corresponding.
PARALLEL_EDGE_GROUPS = [
    [(0, 1), (3, 2), (4, 5), (7, 6)],  # reference x direction
    [(0, 3), (1, 2), (4, 7), (5, 6)],  # reference y direction
    [(0, 4), (1, 5), (2, 6), (3, 7)],  # reference z direction
]

HEX_FACES = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


def make_mesh():
    nodes = []
    sections = []

    for i in range(N):
        theta = 2.0 * math.pi * i / N
        alpha = 0.5 * theta  # half-turn over one complete circuit

        er = (math.cos(theta), math.sin(theta), 0.0)
        ez = (0.0, 0.0, 1.0)

        # Rotating square basis in the normal plane to the centreline.
        b1 = tuple(
            math.cos(alpha) * er[j] + math.sin(alpha) * ez[j]
            for j in range(3)
        )
        b2 = tuple(
            -math.sin(alpha) * er[j] + math.cos(alpha) * ez[j]
            for j in range(3)
        )

        centre = (R * math.cos(theta), R * math.sin(theta), 0.0)

        corners = []
        for s, t in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            p = tuple(
                centre[j] + A * (s * b1[j] + t * b2[j])
                for j in range(3)
            )
            corners.append(len(nodes))
            nodes.append(p)
        sections.append(corners)

    raw_cells = []
    for i in range(N):
        a = sections[i]
        if i < N - 1:
            b = sections[i + 1]
        else:
            # At theta=2*pi the cross-section has rotated by pi.
            # Therefore logical corner j coincides with section-0 corner j+2.
            s0 = sections[0]
            b = [s0[2], s0[3], s0[0], s0[1]]
        raw_cells.append(tuple(a + b))

    # Reverse the square ordering on both end faces of every cell so the
    # trilinear Gmsh hex Jacobian is positive for this geometric embedding.
    perm = [0, 3, 2, 1, 4, 7, 6, 5]
    cells = [tuple(c[i] for i in perm) for c in raw_cells]
    return nodes, cells


def orientation_is_satisfiable(cells):
    """
    Solve x_e xor x_f = parity constraints induced by opposite reference edges.

    Each undirected geometric edge has a binary orientation relative to the
    canonical min(vertex)->max(vertex) direction. Within one reference-parallel
    edge group, all four oriented edge directions must agree. A contradiction
    proves that no global edge orientation satisfies the cell convention.
    """
    graph = defaultdict(list)
    all_edges = set()

    for cell_id, cell in enumerate(cells):
        for group in PARALLEL_EDGE_GROUPS:
            local = []
            for ia, ib in group:
                a, b = cell[ia], cell[ib]
                edge = tuple(sorted((a, b)))
                local_flip = 0 if (a, b) == edge else 1
                local.append((edge, local_flip))
                all_edges.add(edge)

            e0, d0 = local[0]
            for e, d in local[1:]:
                parity = d0 ^ d
                graph[e0].append((e, parity, cell_id))
                graph[e].append((e0, parity, cell_id))

    value = {}
    parent = {}

    for root in all_edges:
        if root in value:
            continue
        value[root] = 0
        queue = deque([root])

        while queue:
            u = queue.popleft()
            for v, parity, cell_id in graph[u]:
                wanted = value[u] ^ parity
                if v not in value:
                    value[v] = wanted
                    parent[v] = (u, parity, cell_id)
                    queue.append(v)
                elif value[v] != wanted:
                    return False, {
                        "edge_a": u,
                        "edge_b": v,
                        "cell": cell_id,
                        "required_parity": parity,
                        "value_a": value[u],
                        "value_b": value[v],
                    }

    return True, None


def boundary_faces(cells):
    seen = {}
    for ci, cell in enumerate(cells):
        for local_face in HEX_FACES:
            face = tuple(cell[i] for i in local_face)
            key = tuple(sorted(face))
            if key in seen:
                seen[key]["count"] += 1
            else:
                seen[key] = {"count": 1, "face": face, "cell": ci}
    return [item["face"] for item in seen.values() if item["count"] == 1]


def write_gmsh(path, nodes, cells):
    bfaces = boundary_faces(cells)

    with open(path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for i, (x, y, z) in enumerate(nodes, 1):
            f.write(f"{i} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")

        f.write("$Elements\n")
        f.write(f"{len(bfaces) + len(cells)}\n")

        eid = 1
        # Gmsh type 3 = 4-node quadrilateral.
        for face in bfaces:
            verts = " ".join(str(v + 1) for v in face)
            f.write(f"{eid} 3 2 1 1 {verts}\n")
            eid += 1

        # Gmsh type 5 = 8-node hexahedron.
        for cell in cells:
            verts = " ".join(str(v + 1) for v in cell)
            f.write(f"{eid} 5 2 1 1 {verts}\n")
            eid += 1

        f.write("$EndElements\n")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "nonorientable_hex_ring.msh"
    )

    nodes, cells = make_mesh()
    ok, conflict = orientation_is_satisfiable(cells)

    print(f"nodes: {len(nodes)}")
    print(f"hexes: {len(cells)}")
    print(f"globally orientable under opposite-edge convention: {ok}")

    if ok:
        raise RuntimeError(
            "Construction unexpectedly became orientable; refusing to write it."
        )

    print("orientation contradiction:", conflict)
    write_gmsh(path, nodes, cells)
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
