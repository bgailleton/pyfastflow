"""Taichi backend pool."""

from ._bucketed_pool import BucketedPool
from .taichi_handle import TaichiDataHandle


class TaichiPool(BucketedPool):
    """Pool that allocates Taichi field handles."""

    _handle_cls = TaichiDataHandle
