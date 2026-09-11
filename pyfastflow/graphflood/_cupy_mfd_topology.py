"""Compatibility bridge for the MFD topology builders now owned by flow."""

from ..flow._cupy_mfd_topology import (
    build_mfd_topology,
    build_ranked_mfd_topology,
    build_receiver_rank,
    build_surface_mfd_topology,
)

__all__ = [
    "build_mfd_topology",
    "build_ranked_mfd_topology",
    "build_receiver_rank",
    "build_surface_mfd_topology",
]
