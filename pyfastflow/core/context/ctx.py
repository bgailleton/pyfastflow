"""
The reserved `ctx` grammar shared by both template surfaces.

A template is written as `def tmpl(ctx, i): return ctx.grad(ctx.z.get(i), i)`
(python) or as CUDA text with `$ctx....$` spans (cupy). Both spellings name
the same shape: a dotted attribute path rooted at `ctx`. `ctx.z.get(i)` reads
slot `z`, `ctx.grad(...)` calls slot `grad`, `ctx.grid.neighbour(i, k)` calls
member `neighbour` of composed slot `grid`. Every reference through ctx is part
of the contract, called or not (see contract.py's module docstring); nothing
after a call is part of the chain.

CTX_PARAM_NAME is the literal name a python template's first parameter must
carry - see contract.py's `extract_python_contract`, which enforces this.
There is no cupy equivalent: a `$ctx....$` span already names it in its text.

`ctx.bk` (RESERVED_BK_NAME, bk.py) is a second piece of reserved grammar, on
the closure (Taichi/Quadrants) python surface only: the backend-intrinsics
namespace (`ctx.bk.sqrt(x)`, ...), recognised structurally by contract.py and
never a slot a template's Contract requires satisfied - see bk.py.

Author: B.G (08/2026)
"""

CTX_PARAM_NAME = "ctx"
"""The reserved first-parameter name every python template must use - see
extract_python_contract in contract.py."""
