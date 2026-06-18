Serialisation
==============

FUSE supports serialising finite element definitions to JSON. This allows elements to be saved, shared, and dynamically loaded, facilitating interoperability between different software components.

Key Classes
-----------

- :class:`fuse.serialisation.ElementSerialiser`: Provides methods for serialising and deserialising `ElementTriple` objects.

Usage
--------

.. literalinclude:: ../../../test/test_serialisation.py
    :language: python3
    :dedent:
    :start-after: [test_serialise 0]
    :end-before: [test_serialise 1]
