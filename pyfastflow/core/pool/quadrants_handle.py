"""Quadrants field-backed DataHandle."""

import quadrants as qd

from ._fields_handle import FieldsBuilderDataHandle


class QuadrantsDataHandle(FieldsBuilderDataHandle):
    """Handle backed by one Quadrants field."""

    _BACKEND_NAME = "quadrants"

    _backend = qd
