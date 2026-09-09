"""
Quadrants backend implementation of DataHandle.

Author: B.G (07/2026)
"""

import quadrants as qd

from ._fields_handle import FieldsBuilderDataHandle


class QuadrantsDataHandle(FieldsBuilderDataHandle):
    """
    DataHandle backed by one Quadrants field.

    Author: B.G (07/2026)
    """

    _BACKEND_NAME = "quadrants"

    _backend = qd
