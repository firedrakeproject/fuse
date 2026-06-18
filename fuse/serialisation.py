"""Serialization module.

This module provides tools for encoding and decoding FUSE objects to and from
JSON representations.
"""

import json
from fuse import *
from fuse.spaces.polynomial_spaces import ConstructedPolynomialSpace
from fuse.spaces.element_sobolev_spaces import ElementSobolevSpace
from fuse.spaces.interpolation_spaces import InterpolationSpace
from fuse.traces import Trace
from fuse.triples import ImmersedDOFs
import sympy as sp


class ElementSerialiser():
    """Serializer class for encoding and decoding FUSE objects to/from JSON.

    Attributes
    ----------
    obj_id_counter : dict
        Counter dictionary to generate unique IDs for each object type.
    seen_objs : dict
        Cache of already-encoded objects to handle references and shared nodes.
    obj_storage : dict
        The object storage map that holds the serialized database of objects.
    obj_types : dict
        Mapping from string names to FUSE class types for decoding.
    """

    def __init__(self):
        """Initialize the ElementSerialiser."""
        self.obj_id_counter = {}
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

    def encode(self, obj):
        """Convert a FUSE object to a JSON-formatted string.

        Parameters
        ----------
        obj : object
            A FUSE class instance to serialize.

        Returns
        -------
        str
            The JSON string representation.
        """
        base_obj = self.encode_traverse(obj)
        self.obj_storage["encoded_obj"] = base_obj
        return json.dumps(self.obj_storage, indent=2)

    def decode(self, obj_str):
        """Convert a JSON string back into a FUSE object.

        Parameters
        ----------
        obj_str : str
            The JSON string representation of the object database.

        Returns
        -------
        object
            The decoded FUSE class instance.
        """
        obj_dict = json.loads(obj_str)
        obj = self.decode_traverse(obj_dict["encoded_obj"], obj_dict)
        return obj

    def encode_traverse(self, obj, path=[]):
        """Recursively traverse and encode an object or container.

        Parameters
        ----------
        obj : object
            The current object/node being serialized.
        path : list, optional
            The traversal path list. Defaults to empty list.

        Returns
        -------
        object
            The serialized object or reference string.
        """
        obj_dict = {}

        if isinstance(obj, list) or isinstance(obj, tuple):
            res_array = [{} for i in range(len(obj))]
            for i in range(len(obj)):
                dfs_res = self.encode_traverse(obj[i], path + [i])
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

    def get_id(self, obj):
        """Get or generate a unique ID for the given object.

        Parameters
        ----------
        obj : object
            The object requiring a unique serialization ID.

        Returns
        -------
        int
            A unique integer identifier.
        """
        obj_name = obj.dict_id()
        if obj_name in self.obj_id_counter.keys():
            obj_id = self.obj_id_counter[obj_name]
            self.obj_id_counter[obj_name] += 1
        else:
            obj_id = 0
            self.obj_id_counter[obj_name] = 1
        return obj_id

    def store_obj(self, obj, name, obj_id, obj_dict, path):
        """Store the encoded representation of an object in the serializer.

        Parameters
        ----------
        obj : object
            The original object.
        name : str
            The dictionary identifier class name.
        obj_id : int
            The unique ID of the object.
        obj_dict : dict
            The dictionary representing the object state.
        path : list
            The serialization traversal path.
        """
        self.seen_objs[obj] = {"id": name + "/" + str(obj_id), "path": path, "dict": obj_dict}
        if name in self.obj_storage.keys():
            self.obj_storage[name][obj_id] = obj_dict
        else:
            self.obj_storage[name] = {obj_id: obj_dict}

    def decode_traverse(self, obj, obj_dict):
        """Recursively traverse and decode dictionary states to concrete objects.

        Parameters
        ----------
        obj : object
            The reference string or value to decode.
        obj_dict : dict
            The full serialization database of objects.

        Returns
        -------
        object
            The decoded python/sympy object or primitive.
        """
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
