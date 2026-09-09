"""
Unit 3: builder slots derived from the template's own contract and signature,
not restated by hand. Structural (freeze only, no device compile), so no
backend init is needed.

Author: B.G (09/2026)
"""

import pytest

from pyfastflow.core.context.builder import GroupBuilder, HelperBuilder, KernelBuilder
from pyfastflow.core.context.contract import ContractError
from pyfastflow.core.context.slot import BuildError, SlotKind


# module-level templates so inspect.getsource can recover them


def _kernel(ctx, z, s):
    for i in z:
        s[i] = ctx.K.get(0) + ctx.DT.get(i) * z[i]


def _helper(ctx, z, i):
    return ctx.NX.get(0) + z[i]


def _inner(ctx, i):
    return ctx.NX.get(i)


def _kernel_bad_call(ctx, z):
    for i in z:
        z[i] = ctx.foo(i)


def _kernel_deadend(ctx, z):
    for i in z:
        z[i] = ctx.g.bogus.get(0)


def _kernel_only_K(ctx, z):
    for i in z:
        z[i] = ctx.K.get(0)


def test_kernel_derives_param_and_data_sets_exactly():
    f = KernelBuilder(_kernel).freeze()
    assert f.slots.names(SlotKind.PARAM) == {"K", "DT"}
    assert f.slots.names(SlotKind.DATA) == {"z", "s"}


def test_helper_does_not_turn_call_args_into_data():
    f = HelperBuilder(_helper).freeze()
    assert f.slots.names(SlotKind.PARAM) == {"NX"}
    assert f.slots.names(SlotKind.DATA) == set()  # z, i are device-call arguments


def test_missing_reports_uncomposed_call_root():
    kb = KernelBuilder(_kernel_bad_call)
    assert kb.missing() == {"foo"}
    with pytest.raises(ContractError, match="foo"):
        kb.freeze()


def test_full_chain_failure_names_deepest_node():
    inner = HelperBuilder(_inner).freeze()  # provides NX only
    kb = KernelBuilder(_kernel_deadend).compose("g", inner)
    with pytest.raises(ContractError, match="dead-ends"):
        kb.freeze()


def test_param_rejects_derived_slots():
    with pytest.raises(BuildError, match="already a template-derived"):
        KernelBuilder(_kernel_only_K).param("K").freeze()


def test_param_declares_extra_slot_the_template_omits():
    # NX is not referenced by the template; param() adds it (a share canonical)
    f = KernelBuilder(_kernel_only_K).param("NX").freeze()
    assert f.slots.names(SlotKind.PARAM) == {"K", "NX"}


def test_data_on_non_signature_name_raises():
    with pytest.raises(BuildError, match="not a DATA argument"):
        KernelBuilder(_kernel_only_K).data("nope", dtype="f32").freeze()


def test_data_attaches_dtype_to_signature_arg():
    f = KernelBuilder(_kernel).data("z", dtype="f32").freeze()
    assert f.slots["z"].dtype == "f32"
    assert f.slots["s"].dtype is None


def test_helper_data_raises():
    # framework misuse -> a BuildError subclass (Unit 7), not TypeError
    with pytest.raises(BuildError):
        HelperBuilder(_helper).data("z", dtype="f32")


def test_group_params_are_all_explicit():
    inner = HelperBuilder(_inner).freeze()
    g = GroupBuilder().param("NX").compose("r", inner).freeze()
    assert g.slots.names(SlotKind.PARAM) == {"NX"}
    assert set(g.children) == {"r"}


def test_constructor_requires_freeze():
    f = KernelBuilder(_kernel).freeze()
    assert f.slots.names(SlotKind.PARAM) == {"K", "DT"}
    assert f.slots.names(SlotKind.DATA) == {"z", "s"}
