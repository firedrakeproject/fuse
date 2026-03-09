from FIAT.functional import Functional
from fuse.utils import sympy_to_numpy
from fuse.traces import TrH1
import numpy as np
import sympy as sp


class Pairing():
    """
    Akin to an inner product, the pairing combines a kernel and an input function
    """

    def __init__(self):
        self.entity = None
        self.orientation = None

    def _to_dict(self):
        o_dict = {"entity": self.entity}
        return o_dict


class DeltaPairing(Pairing):
    """
    The delta pairing allows the evaluation at a single points

    Calling method:
    :param: kernel: Normally a PointKernel
    """

    def __init__(self):
        super(DeltaPairing, self).__init__()

    def __call__(self, kernel, v, cell):
        assert isinstance(kernel, PointKernel)
        return v(*kernel.pt)

    def tabulate(self):
        return 1

    def add_entity(self, entity):
        res = DeltaPairing()
        res.entity = entity
        if self.orientation:
            res = res.permute(self.orientation)
        return res

    def permute(self, g):
        res = DeltaPairing()
        if self.entity:
            res.entity = self.entity
        #    res.entity = self.entity.orient(g)
        if self.orientation is not None:
            g = g * self.orientation
        res.orientation = g
        return res

    def __repr__(self):
        return "{fn}({kernel})"

    def dict_id(self):
        return "Delta"

    def _from_dict(obj_dict):
        new_obj = DeltaPairing()
        new_obj.add_entity(obj_dict["entity"])
        return new_obj


class L2Pairing(Pairing):

    def __init__(self):
        super(L2Pairing, self).__init__()

    def __call__(self, kernel, v, cell):
        Qpts, Qwts = self.entity.quadrature(5)
        return sum([wt*np.dot(kernel(*pt), v(*pt)) for pt, wt in zip(Qpts, Qwts)])

    def tabulate(self):
        bvs = np.array(self.entity.basis_vectors())
        if self.orientation:
            new_bvs = np.array(self.entity.orient(self.orientation).basis_vectors())
            basis_change = np.matmul(np.linalg.inv(new_bvs), bvs)
            return basis_change
        return np.eye(bvs.shape[0])

    def add_entity(self, entity):
        res = L2Pairing()
        res.entity = entity
        if self.orientation:
            res = res.permute(self.orientation)
        return res

    def permute(self, g):
        if g.perm.is_Identity:
            return self

        res = L2Pairing()
        if self.entity:
            res.entity = self.entity
        if self.orientation is not None:
            g = g * self.orientation
        res.orientation = g
        return res

    def __repr__(self):
        if self.orientation is not None:
            return "integral_{}({})({{kernel}} * {{fn}}) dx) ".format(str(self.orientation), str(self.entity))
        return "integral_{}({{kernel}} * {{fn}}) dx) ".format(str(self.entity))

    def dict_id(self):
        return "L2Inner"

    def _from_dict(obj_dict):
        new_obj = L2Pairing()
        new_obj.add_entity(obj_dict["entity"])
        return new_obj


class BaseKernel():

    def __init__(self):
        self.attachment = False

    def permute(self, g):
        raise NotImplementedError("This method should be implemented by the subclass")

    def __repr__(self):
        return "BaseKernel"

    def __call__(self, *args):
        raise NotImplementedError("This method should be implemented by the subclass")


class PointKernel(BaseKernel):

    def __init__(self, x):
        if not isinstance(x, tuple):
            x = (x,)
        self.pt = x
        super(PointKernel, self).__init__()

    def __repr__(self):
        x = list(map(str, list(self.pt)))
        return ','.join(x)

    def degree(self, interpolant_degree):
        return interpolant_degree

    def permute(self, g):
        return PointKernel(g(self.pt))

    def __call__(self, *args):
        return self.pt

    def evaluate(self, Qpts, Qwts, immersed, dim):
        return np.array([self.pt for _ in Qpts]).astype(np.float64), np.ones_like(Qwts), [[tuple()] for pt in Qpts]

    def _to_dict(self):
        o_dict = {"pt": self.pt}
        return o_dict

    def dict_id(self):
        return "PointKernel"

    def _from_dict(obj_dict):
        return PointKernel(tuple(obj_dict["pt"]))


class VectorKernel(BaseKernel):

    def __init__(self, x, g=None):
        self.pt = x
        self.g = g
        super(VectorKernel, self).__init__()

    def __repr__(self):
        if isinstance(self.pt, tuple):
            x = list(map(str, list(self.pt)))
        else:
            x = [str(self.pt)]
        return ','.join(x)

    def degree(self, interpolant_degree):
        return interpolant_degree

    def permute(self, g):
        return VectorKernel(self.pt, g)

    def __call__(self, *args):
        return self.pt

    def evaluate(self, Qpts, Qwts, immersed, dim):
        if isinstance(self.pt, int):
            return Qpts, np.array([wt*self.pt for wt in Qwts]).astype(np.float64), [[(i,) for i in range(dim)] for pt in Qpts]
        if not immersed:
            return Qpts, np.array([wt*self.g(self.pt) for wt in Qwts]).astype(np.float64), [[(i,) for i in range(dim)] for pt in Qpts]
        return Qpts, np.array([wt*immersed(self.g(self.pt))for wt in Qwts]).astype(np.float64), [[(i,) for i in range(dim)] for pt in Qpts]

    def _to_dict(self):
        o_dict = {"pt": self.pt}
        return o_dict

    def dict_id(self):
        return "VectorKernel"

    def _from_dict(obj_dict):
        return VectorKernel(tuple(obj_dict["pt"]))


class PolynomialKernel(BaseKernel):

    def __init__(self, fn, g=None, symbols=[], shape=0):
        self.shape = shape
        try:
            if shape != 0:
                self.fn = [sp.sympify(fn[i]).as_poly() for i in range(shape)]
            else:
                self.fn = sp.sympify(fn)
        except ValueError:
            raise ValueError("Function argument or its components must be able to be interpreted as a sympy polynomial")
        self.g = g
        self.syms = symbols
        super(PolynomialKernel, self).__init__()

    def __repr__(self):
        return str(self.fn)

    def degree(self, interpolant_degree):
        if self.shape != 0:
            return max([self.fn[i].as_poly().total_degree() for i in range(self.shape)]) + interpolant_degree
        if len(self.fn.free_symbols) == 0:  # this should probably be removed
            return interpolant_degree
        return self.fn.as_poly().total_degree() + interpolant_degree

    def permute(self, g):
        # new_fn = self.fn.subs({self.syms[i]: g(self.syms)[i] for i in range(len(self.syms))})
        new_fn = self.fn
        return PolynomialKernel(new_fn, g=g, symbols=self.syms, shape=self.shape)

    def __call__(self, *args):
        if self.shape == 0:
            res = sympy_to_numpy(self.fn, self.syms, args[:len(self.syms)])
        else:
            res = []
            for i in range(self.shape):
                res += [sympy_to_numpy(self.fn[i], self.syms, args[:len(self.syms)])]
        return res

    def evaluate(self, Qpts, Qwts, immersed, dim):
        return Qpts, np.array([wt*self(*self.g(pt)) for pt, wt in zip(Qpts, Qwts)]).astype(np.float64), [[(i,) for i in range(dim)] for pt in Qpts]

    def _to_dict(self):
        o_dict = {"fn": self.fn}
        return o_dict

    def dict_id(self):
        return "PolynomialKernel"

    def _from_dict(obj_dict):
        return PolynomialKernel(obj_dict["fn"])


class ComponentKernel(BaseKernel):

    def __init__(self, comp):
        self.comp = comp
        super(ComponentKernel, self).__init__()

    def __repr__(self):
        return f"[{self.comp}]"

    def degree(self, interpolant_degree):
        return interpolant_degree

    def permute(self, g):
        return self

    def __call__(self, *args):
        return tuple(args[i] if i in self.comp else 0 for i in range(len(args)))
#        return tuple(args[c] for c in self.comp)

    def evaluate(self, Qpts, Qwts, immersed, dim):
        return Qpts, Qwts, [[self.comp] for pt in Qpts]

    def _to_dict(self):
        o_dict = {"comp": self.comp}
        return o_dict

    def dict_id(self):
        return "ComponentKernel"

    def _from_dict(obj_dict):
        return ComponentKernel(obj_dict["comp"])


class DOF():

    def __init__(self, pairing, kernel, entity=None, attachment=None, target_space=None, g=None, immersed=False, generation=None, sub_id=None, cell=None, entity_o=False):
        self.pairing = pairing
        self.kernel = kernel
        self.immersed = immersed
        self.cell_defined_on = entity
        self.attachment = attachment
        self.target_space = target_space
        self.g = g
        self.id = None
        self.sub_id = sub_id
        self.cell = cell
        self.entity_o = entity_o

        if generation is None:
            self.generation = {}
        else:
            self.generation = generation
        if entity is not None:
            self.pairing = self.pairing.add_entity(entity)

    def __call__(self, g, entity_o=False):
        new_generation = self.generation.copy()
        return DOF(self.pairing.permute(g), self.kernel.permute(g), self.cell_defined_on, self.attachment, self.target_space, g, self.immersed, new_generation, self.sub_id, self.cell, entity_o)

    def eval(self, fn, pullback=True):
        return self.pairing(self.kernel, fn, self.cell)

    def tabulate(self, Qpts):
        return self.kernel.tabulate(Qpts)

    def add_context(self, dof_gen, cell, space, g, overall_id=None, generator_id=None):
        # For some of these, we only want to store the first instance of each
        self.generation[cell.dim()] = dof_gen
        self.cell = cell
        if self.cell_defined_on is None:
            self.trace_entity = cell
            self.cell_defined_on = cell
            self.pairing = self.pairing.add_entity(cell)
        if self.target_space is None:
            self.target_space = space
        if self.id is None and overall_id is not None:
            self.id = overall_id
        if self.sub_id is None and generator_id is not None:
            self.sub_id = generator_id

    def convert_to_fiat(self, ref_el, interpolant_degree, value_shape=tuple()):
        # TODO deriv dict needs implementing (currently {})
        return Functional(ref_el, value_shape, self.to_quadrature(interpolant_degree), {}, str(self))

    def to_quadrature(self, arg_degree):
        Qpts, Qwts = self.cell_defined_on.quadrature(self.kernel.degree(arg_degree))
        Qwts = Qwts.reshape(Qwts.shape + (1,))

        if self.immersed and isinstance(self.kernel, VectorKernel):
            def immersed(pt):
                basis = np.array(self.cell_defined_on.basis_vectors()).T
                basis_coeffs = np.matmul(np.linalg.inv(basis), np.array(pt))
                immersed_basis = np.array(self.cell.basis_vectors(entity=self.cell_defined_on))
                return np.matmul(basis_coeffs, immersed_basis)
        else:
            immersed = self.immersed
        pts, wts, comps = self.kernel.evaluate(Qpts, Qwts, immersed, self.cell.dimension)

        if self.immersed:
            # need to compute jacobian from attachment.
            pts = np.array([self.cell.attachment(self.cell.id, self.cell_defined_on.id)(*pt) for pt in pts])
            # if self.pairing.orientation:
            #     immersion = self.target_space.tabulate(wts, self.pairing.entity.orient(self.pairing.orientation))[0]
            # else:
            immersion = self.target_space.tabulate(pts, self.cell_defined_on)
            # Special case - force evaluation on different orientation of entity for construction of matrix transforms
            if self.entity_o:
                immersion = self.target_space.tabulate(wts, self.pairing.entity.orient(self.entity_o))
            if isinstance(self.target_space, TrH1):
                new_wts = wts
            else:
                new_wts = np.outer(wts, immersion)
        else:
            new_wts = wts
        # pt dict is { pt: [(weight, component)]}
        pt_dict = {tuple(pt): [(w, c) for w, c in zip(wt, cp)] for pt, wt, cp in zip(pts, new_wts, comps)}
        if self.cell_defined_on.dimension == 2:
            np.set_printoptions(linewidth=90, precision=4, suppress=True)
            for key, val in pt_dict.items():
                print(np.array(key), ":", np.array([v[0] for v in val]))
        return pt_dict

    def __repr__(self, fn="v"):
        return str(self.pairing).format(fn=fn, kernel=self.kernel)

    def immerse(self, entity, attachment, target_space, g, triple):
        new_generation = self.generation.copy()
        return ImmersedDOF(self.pairing, self.kernel, entity, attachment, target_space, g, triple, new_generation, self.sub_id, self.cell)

    def _to_dict(self):
        """ almost certainly needs more things"""
        o_dict = {"pairing": self.pairing, "kernel": self.kernel}
        return o_dict

    def dict_id(self):
        return "DOF"

    def _from_dict(obj_dict):
        return DOF(obj_dict["pairing"], obj_dict["kernel"])


class ImmersedDOF(DOF):
    # probably need to add a convert to fiat method here to capture derivatives from immersion
    def __init__(self, pairing, kernel, entity=None, attachment=None, target_space=None, g=None, triple=None, generation=None, sub_id=None, cell=None, entity_o=False):
        self.immersed = True
        self.triple = triple
        super(ImmersedDOF, self).__init__(pairing, kernel, entity=entity, attachment=attachment, target_space=target_space, g=g, immersed=True, generation=generation, sub_id=sub_id, cell=cell, entity_o=entity_o)

    def eval(self, fn, pullback=True):
        attached_fn = fn.attach(self.attachment)

        if pullback:
            attached_fn = self.target_space(attached_fn, self.cell_defined_on)

        return self.pairing(self.kernel, attached_fn, self.cell)

    def tabulate(self, Qpts):
        # modify this to take reference space q pts
        immersion = self.target_space.tabulate(Qpts, self.pairing.entity)
        res, _ = self.kernel.tabulate(Qpts, self.attachment)
        return immersion*res

    def __call__(self, g, entity_o=False):
        index_trace = self.cell.d_entities_ids(self.cell_defined_on.dim()).index(self.cell_defined_on.id)
        permuted_e, permuted_g = self.cell.permute_entities(g, self.cell_defined_on.dim())[index_trace]
        new_cell_defined_on = self.cell.get_node(permuted_e)

        new_attach = lambda *x: g(self.attachment(*x))
        return ImmersedDOF(self.pairing.permute(permuted_g), self.kernel.permute(permuted_g), new_cell_defined_on,
                           new_attach, self.target_space, g, self.triple, self.generation, self.sub_id, self.cell, entity_o)

    def __repr__(self):
        fn = "tr_{1}_{0}(v)".format(str(self.cell_defined_on), str(self.target_space))
        return super(ImmersedDOF, self).__repr__(fn)

    def immerse(self, entity, attachment, trace, g):
        raise RuntimeError("Error: Immersing twice not supported")


class FuseFunction():

    def __init__(self, eq, attach_func=None, symbols=None):
        self.eq = eq
        self.attach_func = attach_func
        self.symbols = symbols

    def __call__(self, *x, sym=False):
        if self.symbols:
            if self.attach_func and not sym:
                res = self.eq.subs({symb: val for (symb, val) in zip(self.symbols, self.attach_func(*x))})
            else:
                res = self.eq.subs({symb: val for (symb, val) in zip(self.symbols, x)})
            if res.free_symbols == set():
                array = np.array(res).astype(np.float64)
                return array
            else:
                return res
        if self.attach_func and not sym:
            return self.eq(*self.attach_func(*x))
        else:
            # TODO remove this as will already be symbolic
            return self.eq(*x)

    def attach(self, attachment):
        if not self.attach_func:
            return FuseFunction(self.eq, attach_func=attachment, symbols=self.symbols)
        else:
            old_attach = self.attach_func
            if self.symbols:
                return FuseFunction(self.eq,
                                    attach_func=attachment(old_attach(*self.symbols)),
                                    symbols=self.symbols)
            else:
                return FuseFunction(self.eq,
                                    attach_func=lambda *x: attachment(old_attach(*x)))

    def __repr__(self):
        if self.attach_func:
            return "v(G(x))"
        else:
            return "v(x)"

    def _to_dict(self):
        return {"eq": self.eq}

    def dict_id(self):
        return "Function"
