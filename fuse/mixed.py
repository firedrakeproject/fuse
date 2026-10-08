import numpy as np
import finat.ufl
from FIAT.mixed import MixedElement as FIATMixedElement
from fuse.triples import ElementTriple
from fuse.dof import DeltaPairing, L2Pairing, FuseFunction, PointKernel
from fuse.traces import Trace


class MixedTriple():
    """
    A mixed element: the Cartesian product of a sequence of element triples
    defined on a common cell.

    Unlike an :class:`ElementTriple`, a mixed element is not itself a Ciarlet
    element - it has no single polynomial space or nodal basis, and so does not
    subclass :class:`ElementTriple`. Its degrees of freedom are the disjoint union
    of the sub-element functionals, its value shape is the sum of the sub-element
    value sizes, and each sub-element retains its own mapping. The name follows the
    ``VectorTriple`` / ``TensorProductTriple`` family and avoids clashing with
    ``ufl.MixedElement``.

    :param sub_elements: The element triples to combine. May be passed either as
        separate arguments or as a single iterable.
    """

    def __init__(self, *sub_elements):
        if len(sub_elements) == 1 and not isinstance(sub_elements[0], ElementTriple):
            sub_elements = tuple(sub_elements[0])
        if len(sub_elements) < 2:
            raise ValueError("A mixed element requires at least two sub-elements")
        for e in sub_elements:
            if not isinstance(e, ElementTriple):
                raise ValueError("Mixed element sub-elements must be ElementTriples")

        def cell_signature(cell):
            return (cell.get_spatial_dimension(), len(cell.vertices()))

        ref = cell_signature(sub_elements[0].cell)
        if not all(cell_signature(e.cell) == ref for e in sub_elements):
            raise ValueError("Mixed element sub-elements must be defined on the same cell")

        self._sub_elements = tuple(sub_elements)
        self.cell = sub_elements[0].cell

    @property
    def sub_elements(self):
        return list(self._sub_elements)

    def __repr__(self):
        return "MixedTriple(%s)" % ", ".join(repr(e) for e in self._sub_elements)

    def _sub_value_size(self, e):
        return int(np.prod(e.get_value_shape(), dtype=int))

    def get_value_shape(self):
        return (sum(self._sub_value_size(e) for e in self._sub_elements),)

    def num_dofs(self):
        return sum(e.num_dofs() for e in self._sub_elements)

    @property
    def entity_ids(self):
        """Entity to dof-id map, offset-concatenated across the sub-elements."""
        combined = None
        offset = 0
        for e in self._sub_elements:
            e.to_ufl()
            if combined is None:
                combined = {dim: {ent: [] for ent in e.entity_ids[dim]}
                            for dim in e.entity_ids}
            for dim in e.entity_ids:
                for ent in e.entity_ids[dim]:
                    combined[dim][ent] += [i + offset for i in e.entity_ids[dim][ent]]
            offset += e.num_dofs()
        return combined

    def to_ufl(self):
        return finat.ufl.MixedElement(*[e.to_ufl() for e in self._sub_elements])

    def to_fiat(self):
        return FIATMixedElement([e.to_fiat() for e in self._sub_elements])

    def generate(self):
        dofs = []
        for e in self._sub_elements:
            dofs.extend(e.generate())
        return dofs

    def plot(self, filename="temp.png"):
        import matplotlib.pyplot as plt
        if self.cell.dimension == 0:
            raise ValueError("Dimension 0 cells cannot be plotted")
        if self.cell.dimension > 3:
            raise ValueError("Plotting not supported in this dimension")

        identity = FuseFunction(lambda *x: x)
        fig = plt.figure()
        if self.cell.dimension < 3:
            ax = plt.gca()
            self.cell.plot(show=False, plain=True, ax=ax)
        else:
            ax = fig.add_subplot(projection='3d')
            self.cell.plot3d(show=False, ax=ax)

        for block, e in enumerate(self._sub_elements):
            for dof in e.generate():
                center, color = e.get_dof_info(dof)
                if center is None:
                    center = [0, 0, 0]
                if isinstance(dof.pairing, DeltaPairing) and isinstance(dof.kernel, PointKernel):
                    coord = dof.eval(identity, pullback=False)
                elif isinstance(dof.pairing, L2Pairing):
                    coord = center
                else:
                    coord = center
                if len(coord) == 1:
                    coord = (coord[0], 0)
                if isinstance(dof.target_space, Trace):
                    dof.target_space.plot(ax, coord, dof.cell_defined_on, color=color)
                else:
                    ax.scatter(*coord, color=color)
                ax.text(*coord, "%d.%d" % (block, dof.id))
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if filename:
            fig.savefig(filename)
        else:
            plt.show()

    def _to_dict(self):
        return {"sub_elements": list(self._sub_elements)}

    def dict_id(self):
        return "MixedTriple"

    def _from_dict(o_dict):
        return MixedTriple(o_dict["sub_elements"])
