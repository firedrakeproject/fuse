"""Trace spaces module.

This module defines trace spaces used to represent restrictions of function
spaces to sub-entities of a cell complex.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from fuse.utils import sympy_to_numpy, numpy_to_str_tuple


class Trace():
    """Base class for trace spaces on cell complexes.

    Parameters
    ----------
    cell : Point
        The domain (cell) on which the trace is defined.
    """

    def __init__(self, cell):
        """Initialize the Trace space.

        Parameters
        ----------
        cell : Point
            The domain cell.
        """
        self.domain = cell

    def __call__(self, trace_entity):
        """Evaluate the trace on a sub-entity.

        Parameters
        ----------
        trace_entity : Point
            The sub-entity to trace onto.
        """
        raise NotImplementedError("Trace uninstanitated")

    def plot(self, ax, coord, trace_entity, **kwargs):
        """Plot the degrees of freedom associated with the trace.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis.
        coord : array_like
            Coordinate point of the DOF.
        trace_entity : Point
            The sub-entity of the trace.
        """
        raise NotImplementedError("Trace uninstanitated")

    def tabulate(self, Qwts, trace_entity):
        """Tabulate the trace basis on the entity.

        Parameters
        ----------
        Qwts : array_like
            Quadrature weights or points.
        trace_entity : Point
            The sub-entity.
        """
        raise NotImplementedError("Tabulation uninstantiated")

    def _to_dict(self):
        return {"trace": str(self)}

    def dict_id(self):
        return "Trace"

    def _from_dict(obj_dict):
        # might want to actually save these as functions or something for ambiguity?
        tr_id = obj_dict["trace"]
        if tr_id == "H1":
            return TrH1
        elif tr_id == "HDiv":
            return TrHDiv
        elif tr_id == "HCurl":
            return TrHCurl
        elif tr_id == "Grad":
            return TrGrad
        elif tr_id == "Hess":
            return TrHess
        raise ValueError("Trace not found")


class TrH1(Trace):
    """H1 trace space representation."""

    def __init__(self, cell):
        """Initialize the H1 trace.

        Parameters
        ----------
        cell : Point
            The domain cell.
        """
        super(TrH1, self).__init__(cell)

    def __call__(self, v, trace_entity):
        """Apply H1 trace mapping.

        Parameters
        ----------
        v : callable
            The function to trace.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        callable
            The traced function.
        """
        return v

    def plot(self, ax, coord, trace_entity, **kwargs):
        """Plot the degrees of freedom for H1 trace.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis.
        coord : array_like
            Coordinate point of the DOF.
        trace_entity : Point
            The sub-entity.
        """
        ax.scatter(*coord, **kwargs)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        """Generate TikZ representation for the trace.

        Parameters
        ----------
        coord : array_like
            Coordinate point of the DOF.
        trace_entity : Point
            The sub-entity.
        scale : float
            Scale factor for TikZ.
        color : str, optional
            Color of the TikZ node. Defaults to "black".

        Returns
        -------
        str
            The TikZ command.
        """
        return f"\\filldraw[{color}] {numpy_to_str_tuple(coord, scale)} circle (2pt) node[anchor = south] {{}};"

    def tabulate(self, Qpts, trace_entity):
        """Tabulate the H1 trace on the entity.

        Parameters
        ----------
        Qpts : array_like
            Quadrature points.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        numpy.ndarray
            All ones representing H1 trace DOF scaling.
        """
        return np.ones(len(Qpts))

    def manipulate_basis(self, basis):
        """Manipulate the basis for the H1 trace space.

        Parameters
        ----------
        basis : numpy.ndarray
            The input basis.

        Returns
        -------
        numpy.ndarray
            The manipulated basis.
        """
        return np.array([1])

    def __repr__(self):
        return "H1"


class TrHDiv(Trace):
    """H(div) trace space representation (normal trace)."""

    def __init__(self, cell):
        """Initialize the H(div) trace.

        Parameters
        ----------
        cell : Point
            The domain cell.
        """
        super(TrHDiv, self).__init__(cell)

    def __call__(self, v, trace_entity):
        """Apply H(div) trace mapping (dot product with normal).

        Parameters
        ----------
        v : callable
            The vector-valued function.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        callable
            The scalar-valued trace function (normal component).
        """
        def apply(*x):
            result = np.dot(self.tabulate(None, trace_entity), np.array(v(*x)).squeeze())
            if isinstance(result, np.float64):
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, **kwargs):
        """Plot the normal vector DOF for H(div) trace.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis.
        coord : array_like
            Coordinate point of the DOF.
        trace_entity : Point
            The sub-entity.
        """
        vec = self.tabulate([], trace_entity).squeeze()
        ax.quiver(*coord, *vec, **kwargs)

    def tabulate(self, Qwts, trace_entity):
        """Tabulate normal component on the entity.

        Parameters
        ----------
        Qwts : array_like
            Quadrature weights.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        numpy.ndarray
            The normal vector components.
        """
        cellEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        basis = cellEntityBasis
        if trace_entity.dimension == 1:
            result = np.matmul(basis, np.array([[0, -1], [1, 0]]))
        elif trace_entity.dimension == 2:
            result = np.cross(basis[0], basis[1])
        else:
            raise ValueError("Immersion of HDiv edges not defined in 3D")
        return result

    def manipulate_basis(self, basis):
        """Manipulate the basis vectors for normal component.

        Parameters
        ----------
        basis : numpy.ndarray
            The input basis.

        Returns
        -------
        numpy.ndarray
            The normal vector.
        """
        if basis.shape == (1, 1):
            return basis
        elif basis.shape == (1, 2):
            result = np.matmul(basis, np.array([[0, -1], [1, 0]]))
        elif basis.shape[0] == 2:
            result = np.cross(basis[0], basis[1])
        else:
            raise ValueError("Immersion of HDiv edges not defined in 3D")
        return result

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        """Generate TikZ representation for the H(div) normal vector DOF.

        Parameters
        ----------
        coord : array_like
            Coordinate point of the DOF.
        trace_entity : Point
            The sub-entity.
        scale : float
            Scale factor.
        color : str, optional
            Color. Defaults to "black".

        Returns
        -------
        str
            The TikZ draw command.
        """
        vec = self.tabulate([], trace_entity).squeeze()
        end_point = [coord[i] + 0.25*vec[i] for i in range(len(coord))]
        arw = "-{Stealth[length=3mm, width=2mm]}"
        return f"\\draw[thick, {color}, {arw}] {numpy_to_str_tuple(coord, scale)} -- {numpy_to_str_tuple(end_point, scale)};"

    def __repr__(self):
        return "HDiv"


class TrHCurl(Trace):
    """H(curl) trace space representation (tangential trace)."""

    def __init__(self, cell):
        """Initialize the H(curl) trace.

        Parameters
        ----------
        cell : Point
            The domain cell.
        """
        super(TrHCurl, self).__init__(cell)

    def __call__(self, v, trace_entity):
        """Apply H(curl) trace mapping (projection onto tangent).

        Parameters
        ----------
        v : callable
            The vector-valued function.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        callable
            The tangent component function.
        """
        def apply(*x):
            result = np.dot(self.tabulate(None, trace_entity), np.array(v(*x)).squeeze())
            if isinstance(result, np.float64):
                return (result,)
            return tuple(result)
        return apply

    def tabulate(self, Qwts, trace_entity):
        """Tabulate tangent component on the entity.

        Parameters
        ----------
        Qwts : array_like
            Quadrature weights.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        numpy.ndarray
            The tangent vector.
        """
        subEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        return subEntityBasis

    def manipulate_basis(self, basis):
        """Extract tangent component from the basis.

        Parameters
        ----------
        basis : numpy.ndarray
            The input basis.

        Returns
        -------
        numpy.ndarray
            The tangent vector.
        """
        return basis[0]

    def plot(self, ax, coord, trace_entity, **kwargs):
        """Plot the tangential vector DOF for H(curl) trace.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis.
        coord : array_like
            Coordinate point.
        trace_entity : Point
            The sub-entity.
        """
        vec = self.tabulate([], trace_entity).squeeze()
        ax.quiver(*coord, *vec, **kwargs)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        """Generate TikZ representation for the H(curl) tangent vector DOF.

        Parameters
        ----------
        coord : array_like
            Coordinate point.
        trace_entity : Point
            The sub-entity.
        scale : float
            Scale factor.
        color : str, optional
            Color. Defaults to "black".

        Returns
        -------
        str
            The TikZ draw command.
        """
        vec = self.tabulate([], trace_entity).squeeze()
        end_point = [coord[i] + 0.25*vec[i] for i in range(len(coord))]
        arw = "-{Stealth[length=3mm, width=2mm]}"
        return f"\\draw[thick, {color}, {arw}] {numpy_to_str_tuple(coord, scale)} -- {numpy_to_str_tuple(end_point, scale)};"

    def __repr__(self):
        return "HCurl"


class TrGrad(Trace):
    """Gradient trace space representation."""

    def __init__(self, cell):
        """Initialize the gradient trace.

        Parameters
        ----------
        cell : Point
            The domain cell.
        """
        super(TrGrad, self).__init__(cell)

    def __call__(self, v, trace_entity):
        """Apply gradient trace mapping.

        Parameters
        ----------
        v : callable
            The function.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        callable
            The gradient trace function.
        """
        raise NotImplementedError("Gradient immersions are under development")
        g = None
        tangent = np.array(g(np.array(self.domain.basis_vectors())[0]))

        def apply(*x):
            X = sp.DeferredVector('x')
            dX = tuple([X[i] for i in range(self.domain.dim())])
            compute_v = v(*dX, sym=True)
            grad_v = sp.Matrix([sp.diff(compute_v, dX[i]) for i in range(len(dX))])
            eval_grad_v = sympy_to_numpy(grad_v, dX, v.attach_func(*x))
            result = np.dot(tangent, np.array(eval_grad_v))

            if not hasattr(result, "__iter__"):
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, **kwargs):
        """Plot the gradient trace DOF (represented as a circle).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis.
        coord : array_like
            Coordinate point.
        trace_entity : Point
            The sub-entity.
        """
        circle1 = plt.Circle(coord, 0.075, fill=False, **kwargs)
        ax.add_patch(circle1)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        """Generate TikZ representation for gradient trace DOF.

        Parameters
        ----------
        coord : array_like
            Coordinate point.
        trace_entity : Point
            The sub-entity.
        scale : float
            Scale factor.
        color : str, optional
            Color. Defaults to "black".

        Returns
        -------
        str
            The TikZ draw command.
        """
        return f"\\draw[{color}] {numpy_to_str_tuple(coord, scale)} circle (4pt) node[anchor = south] {{}};"

    def __repr__(self):
        return "Grad"


class TrHess(Trace):
    """Hessian trace space representation."""

    def __init__(self, cell):
        """Initialize the Hessian trace.

        Parameters
        ----------
        cell : Point
            The domain cell.
        """
        super(TrHess, self).__init__(cell)

    def __call__(self, v, trace_entity):
        """Apply Hessian trace mapping.

        Parameters
        ----------
        v : callable
            The function.
        trace_entity : Point
            The sub-entity.

        Returns
        -------
        callable
            The Hessian trace function.
        """
        raise NotImplementedError("Hessian trace needs reviewing")
        g = None
        b0, b1 = self.domain.basis_vectors()
        tangent0 = np.array(g(b0))
        tangent1 = np.array(g(b1))

        def apply(*x):
            X = sp.DeferredVector('x')

            dX = tuple([X[i] for i in range(self.domain.dim())])
            hess_v = sp.Matrix([[sp.diff(v(*dX, sym=True), dX[i], dX[j]) for i in range(len(dX))] for j in range(len(dX))])
            eval_hess_v = sympy_to_numpy(hess_v, dX, v.attach_func(*x))
            result = np.dot(np.matmul(tangent0, np.array(eval_hess_v)), tangent1)
            if not hasattr(result, "__iter__"):
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, **kwargs):
        """Plot Hessian trace DOF (represented as a circle).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis.
        coord : array_like
            Coordinate point.
        trace_entity : Point
            The sub-entity.
        """
        circle1 = plt.Circle(coord, 0.15, fill=False, **kwargs)
        ax.add_patch(circle1)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        """Generate TikZ representation for Hessian trace DOF.

        Parameters
        ----------
        coord : array_like
            Coordinate point.
        trace_entity : Point
            The sub-entity.
        scale : float
            Scale factor.
        color : str, optional
            Color. Defaults to "black".

        Returns
        -------
        str
            The TikZ draw command.
        """
        return f"\\draw[{color}] {numpy_to_str_tuple(coord, scale)} circle (6pt) node[anchor = south] {{}};"

    def __repr__(self):
        return "Hess"
