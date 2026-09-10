"""CuPy backend pool."""

from ._bucketed_pool import BucketedPool
from .cupy_handle import CupyDataHandle


class CupyPool(BucketedPool):
    """Pool that allocates CuPy array handles."""

    _handle_cls = CupyDataHandle
