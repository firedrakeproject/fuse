from firedrake import *
from fuse import *
import numpy as np
from test_2d_examples_docs import construct_cg3
from test_convert_to_fiat import create_cr, create_cg1


def create_cr3(cell):
    Pk = PolynomialSpace(3)
    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), [DOFGenerator([DOF(DeltaPairing(), PointKernel((-np.sqrt(3/5),)))], S2, S1),
                                                                               DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)])
    edge_xs = [immerse(cell, edge_dg0, TrH1)]
    center = [DOF(DeltaPairing(), PointKernel((0, 0)))]

    return ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(edge_xs, C3, S1), DOFGenerator(center, S1, S1)])


errors_cg = []
errors_cr = []
target_eigenvalue = 9

# exact eigenvalues are n^2 + m^2, n, m \in \mathbb{N}
exact_eigenvalues = [2, 5, 5, 8, 10, 10, 13, 13, 17, 17, 18, 20, 20, 25, 25, 26, 26]

cg3 = construct_cg3(polygon(3))
cr3 = create_cr3(polygon(3))
cr1 = create_cr(polygon(3))
cg1 = create_cg1(polygon(3))

for N in [50, 100, 200]:
    mesh = RectangleMesh(N, N, pi, pi, use_fuse=True)

    for elem, space in zip([cg3, cr3], ["CG", "CR"]):
        V = FunctionSpace(mesh, elem.to_ufl_elem())
        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx
        b = inner(u, v)*dx
        bc = DirichletBC(V, 0, "on_boundary")
        eigenproblem = LinearEigenproblem(a, b, bc)

        sp = {"eps_gen_hermitian": None,  # kind of problem
              "eps_smallest_real": None,  # which eigenvalues
              "eps_monitor": None,        # monitor
              "eps_type": "krylovschur"}  # algorithm

        # request ten eigenvalues
        eigensolver = LinearEigensolver(eigenproblem, 10, solver_parameters=sp)
        nconv = eigensolver.solve()  # number of converged eigenvalues

        # Take real part, since we know it is Hermitian
        eigenvalues = [eigensolver.eigenvalue(i).real for i in range(nconv)]
        print(f"{space}/{N}. Eigenvalues: ", eigenvalues)
        # Only take real part; .eigenfunction returns (real, complex)
        eigenfuncs = [eigensolver.eigenfunction(i)[0] for i in range(nconv)]
        h = 1/N
        k = 0.1932
        print("magic number ", [e/(1+k**2*e*h) for e in eigenvalues])
        if space == "CR":
            errors_cr.append(eigenvalues[target_eigenvalue] - exact_eigenvalues[target_eigenvalue])
        elif space == "CG":
            errors_cg.append(eigenvalues[target_eigenvalue] - exact_eigenvalues[target_eigenvalue])


convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])
print("Convergence orders (CG)", convergence_orders(errors_cg))
print("Convergence orders (CR)", convergence_orders(errors_cr))
