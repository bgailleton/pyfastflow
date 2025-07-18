"""
GraphFlood algorithms submodule for FastFlow Taichi.

Contains all shallow water flow algorithms and utilities:

- gf_fields: Field management and data structures for flow variables
- gf_hydrodynamics: Core hydrodynamic computation kernels with Manning's friction
- Flooder: High-level interface for flood modeling workflows

Key Features:
- 2D shallow water equations with Manning's friction
- GPU-accelerated flood propagation using Taichi
- Integration with FastFlow routing for initial conditions
- Precipitation and boundary condition support
- Configurable hydrodynamic parameters

Usage:
    import pyfastflow as pf
    
    # Initialize flow router first
    router = pf.flow.FlowRouter(nx=100, ny=100, dx=1.0)
    
    # Create flood model
    flooder = pf.graphflood.Flooder(
        router=router,
        precipitation_rates=10e-3/3600,  # m/s
        manning=1e-3,                    # Manning's n
        edge_slope=1e-2                  # Boundary slope
    )

Author: B.G.
"""

# Import all graphflood modules - accessible as pf.graphflood.module_name
from .gf_fields import *
from .gf_hydrodynamics import *

# Export all modules
__all__ = [
    # Core classes
    "Flooder",
    
    # Individual functions
    "diffuse_Q_constant_prec",
    "graphflood_core_cte_mannings",
    
    # Module names
    "gf_fields",
    "gf_hydrodynamics"
]