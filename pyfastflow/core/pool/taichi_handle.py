"""Taichi field-backed DataHandle."""

import taichi as ti

from ._fields_handle import FieldsBuilderDataHandle


class TaichiDataHandle(FieldsBuilderDataHandle):
    """Handle backed by one Taichi field."""

    _BACKEND_NAME = "taichi"

    _backend = ti
