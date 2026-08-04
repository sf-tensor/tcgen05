import pytest

from tcgen05_model import DotProduct, make_model, mma, mma_dot, sparse_active_indices
from tcgen05_model.validation import validate


def test_frozen_b200_validation_vectors():
    report = validate()
    assert report.passed, report.failures
    assert report.checked >= 50


def test_tf32_matrix_example():
    a = ((0x3F800000, 0x40000000),)
    b = ((0x40000000,), (0x3F800000,))
    assert mma(a, b, ((0,),), a_format="tf32") == ((0x40800000,),)


def test_integer_saturation():
    model = make_model("u8", "u8", d_type="s32", saturate=True)
    case = DotProduct((0xFF,) * 32, (0xFF,) * 32, 0x7FFFFFF0)
    assert model.eval(case) == 0x7FFFFFFF


def test_sparse_metadata_repeats_for_long_k():
    active = sparse_active_indices("e4m3", 64, 0x44444444)
    assert active == frozenset((*range(0, 32, 4), *range(1, 32, 4), *range(32, 64, 4), *range(33, 64, 4)))


def test_sparse_disabled_nan_is_not_classified():
    a = (0x3F800000,) * 16
    b = tuple(0x3F800000 if k % 2 == 0 else 0x7FC00000 for k in range(16))
    assert mma_dot(a, b, a_format="tf32", sparse_metadata=0x44444444) == 0x41000000


def test_rejects_invalid_scale_combination():
    with pytest.raises(ValueError):
        make_model("e4m3", scaling="ue4m3")
