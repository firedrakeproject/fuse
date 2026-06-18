:orphan: true

.. title:: FUSE

FUSE: A finite, unified, serialisable finite element.
======================================================

FUSE (Finite Unified Serialisable Element) is a software package designed to automate the implementation of finite elements from their mathematical data. 
Core Goals
----------

- **Automated Implementation:** Generate finite element implementations directly from data.
- **Unified Definition:** Provide a constructive definition (Expanded Triple) that covers a wide range of elements.
- **Serialisation:** Support JSON serialisation for dynamic element definition and interoperability.
- **Integration:** Work seamlessly with libraries like Firedrake and Basix.

FUSE introduces an expanded definition of finite elements that explicitly handles cell complexes, symmetry groups, and DOF generation.
