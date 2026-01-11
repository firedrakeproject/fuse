import json
from fuse import *
from fuse.spaces.polynomial_spaces import ConstructedPolynomialSpace
from fuse.spaces.element_sobolev_spaces import ElementSobolevSpace
from fuse.spaces.interpolation_spaces import InterpolationSpace
from fuse.traces import Trace
from fuse.triples import ImmersedDOFs
from typing import Any
import sympy as sp

type Terminal = Union[int, str, bool, None]
type RecursionType = Union[FuseType, list[Any], sp.Expr, sp.core.containers.Tuple]
type ObjDictType = dict[str, FuseType | Terminal | list[Any]]


class ElementSerialiser():
    """
    This class provides encoding and decoding to json for objects in the
    fuse language.

    Methods
    --------

    encode: obj: fuse class
        - converts object to a json representation
    decode: obj_str: String
        - converts json string to obj
    """

    def __init__(self):
        self.obj_id_counter: dict[str,int] = {}
        self.seen_objs = {}
        self.obj_storage = {}

        self.obj_types = {"Cell": Point,
                          "Edge": Edge,
                          "Triple": ElementTriple,
                          "Group": GroupRepresentation,
                          "SobolevSpace": ElementSobolevSpace,
                          "InterpolationSpace": InterpolationSpace,
                          "PolynomialSpace": PolynomialSpace,
                          "ConstructedPolynomialSpace": ConstructedPolynomialSpace,
                          "DOF": DOF,
                          "ImmersedDOF": ImmersedDOFs,
                          "DOFGen": DOFGenerator,
                          "Delta": DeltaPairing,
                          "L2Inner": L2Pairing,
                          "PolynomialKernel": PolynomialKernel,
                          "PointKernel": PointKernel,
                          "Trace": Trace
                          }

    def encode(self, obj: FuseType) -> str:
        base_obj = self.encode_traverse(obj)
        self.obj_storage["encoded_obj"] = base_obj
        return json.dumps(self.obj_storage, indent=2)

    def decode(self, obj_str: str) -> FuseType:
        obj_dict = json.loads(obj_str)
        obj = self.decode_traverse(obj_dict["encoded_obj"], obj_dict)
        return obj

    def encode_traverse(self, obj: RecursionType, path: list[str | int] = []) -> FuseType | Terminal | list[Any]:
        obj_dict = {}

        if isinstance(obj, list) or isinstance(obj, tuple):
            res_array: list[RecursionType] = [None for i in range(len(obj))]
            for i in range(len(obj)):
                dfs_res = self.encode_traverse(obj[i], path + [i])
                breakpoint()
                res_array[i] = dfs_res
            return res_array

        if obj in self.seen_objs.keys():
            return self.seen_objs[obj]["id"]

        if hasattr(obj, "_to_dict"):
            for (key, val) in obj._to_dict().items():
                obj_dict[key] = self.encode_traverse(val, path + [key])
            obj_id = self.get_id(obj)
            self.store_obj(obj, obj.dict_id(), obj_id, obj_dict, path)
            return obj.dict_id() + "/" + str(obj_id)

        if isinstance(obj, sp.core.containers.Tuple) or isinstance(obj, sp.Expr):
            return "Sympy/" + sp.srepr(obj)

        return obj

    def get_id(self, obj: FuseType) -> int:
        obj_name = obj.dict_id()
        if obj_name in self.obj_id_counter.keys():
            obj_id = self.obj_id_counter[obj_name]
            self.obj_id_counter[obj_name] += 1
        else:
            obj_id = 0
            self.obj_id_counter[obj_name] = 1
        return obj_id

    def store_obj(self, obj: FuseType, name: str, obj_id: int, obj_dict: ObjDictType, path: list[str | int]):
        self.seen_objs[obj] = {"id": name + "/" + str(obj_id), "path": path, "dict": obj_dict}
        if name in self.obj_storage.keys():
            self.obj_storage[name][obj_id] = obj_dict
        else:
            self.obj_storage[name] = {obj_id: obj_dict}

    def decode_traverse(self, obj: FuseType | str | list[Any] | tuple[Any], obj_dict: ObjDictType):

        if isinstance(obj, str):
            split_str = obj.split("/")
            if split_str[0] in self.obj_types.keys():
                name, obj_id = split_str[0], split_str[1]
                sub_dict = obj_dict[name][obj_id]
                for (key, value) in sub_dict.items():
                    sub_dict[key] = self.decode_traverse(value, obj_dict)
                return self.obj_types[name]._from_dict(sub_dict)
            elif split_str[0] == "Sympy":
                return sp.parse_expr(" ".join(split_str[1:]))
            else:
                return obj
        elif isinstance(obj, list) or isinstance(obj, tuple):
            res_array = [0 for i in range(len(obj))]
            for i in range(len(obj)):
                dfs_res = self.decode_traverse(obj[i], obj_dict)
                res_array[i] = dfs_res
            return res_array

        return obj
