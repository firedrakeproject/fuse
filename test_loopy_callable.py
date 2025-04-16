import numpy as np
# from constantdict import constantdict

import loopy as lp
from loopy.diagnostic import LoopyError
from loopy.target.c import CTarget
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


# {{{ blas callable

# class CBLASGEMV(lp.ScalarCallable):
#     def with_types(self, arg_id_to_dtype, callables_table):
#         mat_dtype = arg_id_to_dtype.get(0)
#         vec_dtype = arg_id_to_dtype.get(1)

#         if mat_dtype is None or vec_dtype is None:
#             # types aren't specialized enough to be resolved
#             return self, callables_table

#         if mat_dtype != vec_dtype:
#             raise LoopyError("GEMV requires same dtypes for matrix and "
#                              "vector")

#         if vec_dtype.numpy_dtype == np.float32:
#             name_in_target = "cblas_sgemv"
#         elif vec_dtype. numpy_dtype == np.float64:
#             name_in_target = "cblas_dgemv"
#         else:
#             raise LoopyError("GEMV is only supported for float32 and float64 "
#                              "types")

#         return (self.copy(name_in_target=name_in_target,
#                           arg_id_to_dtype={
#                               0: vec_dtype,
#                               1: vec_dtype,
#                               -1: vec_dtype}),
#                 callables_table)

#     def with_descrs(self, arg_id_to_descr, callables_table):
#         mat_descr = arg_id_to_descr.get(0)
#         vec_descr = arg_id_to_descr.get(1)
#         res_descr = arg_id_to_descr.get(-1)

#         if mat_descr is None or vec_descr is None or res_descr is None:
#             # shapes aren't specialized enough to be resolved
#             return self, callables_table

#         assert mat_descr.shape[1] == vec_descr.shape[0]
#         assert mat_descr.shape[0] == res_descr.shape[0]
#         assert len(vec_descr.shape) == len(res_descr.shape) == 1
#         # handling only the easy case when stride == 1
#         assert vec_descr.dim_tags[0].stride == 1
#         assert mat_descr.dim_tags[1].stride == 1
#         assert res_descr.dim_tags[0].stride == 1

#         return self.copy(arg_id_to_descr=arg_id_to_descr), callables_table

#     def emit_call_insn(self, insn, target, expression_to_code_mapper):
#         from pymbolic import var
#         mat_descr = self.arg_id_to_descr[0]
#         m, n = mat_descr.shape
#         ecm = expression_to_code_mapper
#         mat, vec, result = insn.expression.parameters
#         # result, = insn.assignees

#         c_parameters = [var("CblasRowMajor"),
#                         var("CblasNoTrans"),
#                         m, n,
#                         1,
#                         ecm(mat).expr,
#                         m,
#                         ecm(vec).expr,
#                         1,
#                         1,
#                         ecm(result).expr,
#                         1]
#         return (var(self.name_in_target)(*c_parameters),
#                 False  # cblas_gemv does not return anything
#                 )

#     def generate_preambles(self, target):
#         assert isinstance(target, CTarget)
#         yield ("99_cblas", "#include <cblas.h>")
#         return

# n = 10
# # 
# transform_insn = lp.CInstruction(tuple(),"""switch(orientation) {case 0:printf("hi\n");break;case 4:printf("hi2\n");break;} """)
# string = "y[:] = gemv(A[:, :], x[:]);"
# # string = "if (orientation==4) { gemv(A[:, :], x[:], y[:]);}"
# # string = "if(orientation=0, gemv(A[:, :], x[:], y[:]),  gemv(A[:, :], x[:], y[:]))"
# knl = lp.make_kernel(
#         tuple(),
#         string, [
#             lp.GlobalArg("A", dtype=np.float64, shape=(n, n)),
#             lp.GlobalArg("x", dtype=np.float64, shape=(n, )),
#             lp.GlobalArg("y", dtype=np.float64, shape=(n, )), ...],
#         target=CTarget())
# # 
#             # lp.GlobalArg("orientation", dtype=np.int8),
# # knl = lp.register_callable(knl, "gemv", CBLASGEMV(name="gemv"))
# print(lp.generate_code_v2(knl).device_code())

class CBLASGEMV(lp.ScalarCallable):
    def with_types(self, arg_id_to_dtype, callables_table):
        mat_dtype = arg_id_to_dtype.get(0)
        vec_dtype = arg_id_to_dtype.get(1)

        if mat_dtype is None or vec_dtype is None:
            # types aren't specialized enough to be resolved
            return self, callables_table

        if mat_dtype != vec_dtype:
            raise LoopyError("GEMV requires same dtypes for matrix and "
                             "vector")

        if vec_dtype.numpy_dtype == np.float32:
            name_in_target = "cblas_sgemv"
        elif vec_dtype. numpy_dtype == np.float64:
            name_in_target = "cblas_dgemv"
        else:
            raise LoopyError("GEMV is only supported for float32 and float64 "
                             "types")

        return (self.copy(name_in_target=name_in_target,
                          arg_id_to_dtype=dict({
                              0: vec_dtype,
                              1: vec_dtype,
                              -1: vec_dtype})),
                callables_table)

    def with_descrs(self, arg_id_to_descr, callables_table):
        mat_descr = arg_id_to_descr.get(0)
        vec_descr = arg_id_to_descr.get(1)
        res_descr = arg_id_to_descr.get(-1)

        if mat_descr is None or vec_descr is None or res_descr is None:
            # shapes aren't specialized enough to be resolved
            return self, callables_table

        assert mat_descr.shape[1] == vec_descr.shape[0]
        assert mat_descr.shape[0] == res_descr.shape[0]
        assert len(vec_descr.shape) == len(res_descr.shape) == 1
        # handling only the easy case when stride == 1
        assert vec_descr.dim_tags[0].stride == 1
        assert mat_descr.dim_tags[1].stride == 1
        assert res_descr.dim_tags[0].stride == 1

        return self.copy(arg_id_to_descr=arg_id_to_descr), callables_table

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        from pymbolic import var
        mat_descr = self.arg_id_to_descr[0]
        m, n = mat_descr.shape
        ecm = expression_to_code_mapper
        mat, vec = insn.expression.parameters
        result, = insn.assignees

        c_parameters = [var("CblasRowMajor"),
                        var("CblasNoTrans"),
                        m, n,
                        1,
                        ecm(mat).expr,
                        1,
                        ecm(vec).expr,
                        1,
                        ecm(result).expr,
                        1]
        return (var(self.name_in_target)(*c_parameters),
                False  # cblas_gemv does not return anything
                )

    def generate_preambles(self, target):
        assert isinstance(target, CTarget)
        yield ("99_cblas", "#include <cblas.h>")
        return

# }}}


# n = 10

# knl = lp.make_kernel(
#         "{:}",
#         """
#         if (orientation = 1, y[:] = gemv(A[:, :], x[:], y[:] = gemv(A[:, :], x[:])
#         """, [
#             lp.GlobalArg("A", dtype=np.float64, shape=(n, n)),
#             lp.GlobalArg("x", dtype=np.float64, shape=(n, )),
#             lp.GlobalArg("orientation", dtype=np.int8),
#             lp.GlobalArg("y", shape=(n, )), ...],
#         target=CTarget())

# knl = lp.register_callable(knl, "gemv", CBLASGEMV(name="gemv"))
# print(lp.generate_code_v2(knl).device_code())
# nint i, j;\nfor(i=0;i<m;i++){\nfor(j=0;j<m;j++){\n res[j] += a[i][j]*b[i];\n}}\nreturn (double *)res;}\n"
    
# matmul_knl = lp.make_function("{[i,j]: 0< i,j < 3}",
#                             """
#                             res[j] =  res[j] + a[i, j]*b[i]
#                             """, name="matmul")

#                                     # transform_insn = lp.CInstruction(tuple(), matrix_switch_statement(os))
#         # dim = os[0].shape[0]
# dim = 3
# args = [lp.GlobalArg("input", dtype=np.float32, shape=(dim, )), lp.GlobalArg("output", dtype=np.dtype(np.float32)),
#         lp.GlobalArg("orientation", dtype=np.uint8)]
# args += [lp.TemporaryVariable(f"mat1", initializer=np.array([[1,2,3],[1,2,3],[1,2,3]]), read_only=True)]
# loopy_knl = lp.make_kernel(
#     domains = "{:}",
#     instructions = "matmul(m, mat1, input, output)",
#     kernel_data=args,
#     target=lp.CTarget())
# knl = lp.merge([matmul_knl, loopy_knl])

# print(lp.generate_code_v2(loopy_knl).device_code())
def matrix_switch_statement(os):
    string = []
    dim = os[0].shape[0]
    string += f"int m={dim};\nswitch (orientation) {{ \n"
    for val in os.keys():
            string += f"case {val}:\n a = mat{val};break;\n"
            # string += f"case {val}:\n output[:] = gemv(mat{val}[:, :], input[:]);break;\n"
    string += "default:\nbreak;\n }\n}"

    return "".join(string)
os = {0: np.array([[2,3,4],[3,5,6],[6,7,8]]), 1: np.array([[5,3,6],[8,9,6],[8,7,8]])}
n = 3

child_knl = lp.make_function(
        "{[i, j]:0<=i, j < 3}",
        """
        res[j] =  res[j] + a[i, j]*b[i]
        """, name="matmul",target=lp.CTarget())
args = [lp.GlobalArg("b", dtype=np.float32, shape=(n, )), lp.TemporaryVariable("a", dtype=np.float32, shape =(n, n)), lp.GlobalArg("res", dtype=np.float32, shape =(n,)),
        lp.ValueArg("o", dtype=np.int8), lp.GlobalArg("a2", dtype=np.float32, shape =(n, n))]
mats = ["o"]
inst = []
string = [f"\nswitch (o) {{ \n"]
for val in sorted(os.keys()):
    string += f"case {val}:\n a2 = mat{val};break;\n"
    inst += [f"""if (o == {val}){{ \n a2 = mat{val};}}\n"""]
    mats += [f"mat{val}"]
    args += [lp.TemporaryVariable(f"mat{val}", initializer=np.array(os[val], dtype=np.float32), dtype=np.float32, read_only=True, address_space=lp.AddressSpace(1))]
string += "default:\nbreak;\n }"
transform_insn = lp.CInstruction(tuple(), "".join(string), assignees=("a2"), read_variables=frozenset(mats), id="assign")

parent_knl = lp.make_kernel(
        "{:}",
        [transform_insn, "res[:] = matmul(a2, b, res) {dep=assign}"],
        kernel_data=args
        ,target=lp.CTarget())
knl = lp.merge([parent_knl, child_knl])
print(knl)
print(knl["loopy_kernel"].stringify(with_dependencies=True))
print(lp.generate_code_v2(knl).device_code())