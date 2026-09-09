"""
Taichi backend implementation of DataHandle.

Author: B.G (07/2026)
"""

import taichi as ti

from ._fields_handle import FieldsBuilderDataHandle


class TaichiDataHandle(FieldsBuilderDataHandle):
    """
    DataHandle backed by one Taichi field.

    Author: B.G (07/2026)
    """

    _BACKEND_NAME = "taichi"

    _backend = ti
