"""Quadrants backend pool."""

from ._bucketed_pool import BucketedPool
from .quadrants_handle import QuadrantsDataHandle


class QuadrantsPool(BucketedPool):
    """Pool that allocates Quadrants field handles."""

    _handle_cls = QuadrantsDataHandle
