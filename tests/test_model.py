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


@pytest.mark.parametrize(
    ("a_format", "b_format", "d_type", "scaling", "kind", "scale_vec"),
    (
        ("tf32", "tf32", "f16", None, None, None),
        ("tf32", "e2m1", "f32", None, None, None),
        ("bf16", "bf16", "f16", None, None, None),
        ("tf32", "tf32", "banana", None, None, None),
        ("e2m1", "e2m1", "f32", "ue8m0", "mxf4", 4),
        ("e4m3", "e4m3", "f32", "ue8m0", "mxf8f6f4", 99),
    ),
)
def test_rejects_unsupported_descriptors(a_format, b_format, d_type, scaling, kind, scale_vec):
    with pytest.raises(ValueError):
        make_model(
            a_format,
            b_format,
            d_type=d_type,
            scaling=scaling,
            kind=kind,
            scale_vec=scale_vec,
        )


def test_accepts_supported_mixed_f8f6f4_descriptor():
    make_model("e2m1", "e4m3", d_type="f16")
    make_model("e2m1", "e4m3", scaling="ue8m0")


def test_integer_descriptor_rejects_block_scale_fields():
    with pytest.raises(ValueError, match="does not use block scaling"):
        make_model("u8", "s8", d_type="s32", scaling="ue8m0")


def test_sparse_mxf8f6f4_uses_physical_k_scale_blocks():
    metadata = (0x44444444, 0x44444444)
    assert mma_dot(
        (0x38,) * 64,
        (0x38,) * 64,
        a_format="e4m3",
        scaling="ue8m0",
        kind="mxf8f6f4",
        scale_vec=1,
        scale_a=(0x7F, 0x80, 0x7F, 0x7F),
        scale_b=(0x7F,) * 4,
        sparse_metadata=metadata,
    ) == 0x42000000


@pytest.mark.parametrize(
    ("scaling", "kind", "scale_vec", "scale"),
    (
        ("ue8m0", "mxf4", 2, 0x7F),
        ("ue8m0", "mxf4nvf4", 2, 0x7F),
        ("ue8m0", "mxf4nvf4", 4, 0x7F),
        ("ue4m3", "mxf4nvf4", 4, 0x38),
    ),
)
def test_sparse_mxf4_families_use_pairwise_4_of_8(scaling, kind, scale_vec, scale):
    a = tuple(0x2 if k % 8 in (2, 3) else 0 for k in range(128))
    assert mma_dot(
        a,
        (0x2,) * 128,
        a_format="e2m1",
        scaling=scaling,
        kind=kind,
        scale_vec=scale_vec,
        scale_a=(scale,) * 4,
        scale_b=(scale,) * 4,
        sparse_metadata=(0x44444444, 0x44444444),
    ) == 0x42000000


@pytest.mark.parametrize(
    ("a_format", "scaling", "kind", "scale_vec", "k", "scale"),
    (
        ("e4m3", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e5m2", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e2m3", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e3m2", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e2m1", "ue8m0", "mxf8f6f4", 1, 32, 0x7F),
        ("e2m1", "ue8m0", "mxf4", 2, 64, 0x7F),
        ("e2m1", "ue8m0", "mxf4nvf4", 2, 64, 0x7F),
        ("e2m1", "ue8m0", "mxf4nvf4", 4, 64, 0x7F),
        ("e2m1", "ue4m3", "mxf4nvf4", 4, 64, 0x38),
    ),
)
def test_block_scaled_zero_canonicalizes_negative_zero(a_format, scaling, kind, scale_vec, k, scale):
    assert mma_dot(
        (0,) * k,
        (0,) * k,
        0x80000000,
        a_format=a_format,
        scaling=scaling,
        kind=kind,
        scale_vec=scale_vec,
        scale_a=(scale,) * 4,
        scale_b=(scale,) * 4,
    ) == 0


@pytest.mark.parametrize(
    ("a_format", "a", "b", "expected"),
    [
        (
            "bf16",
            [0x18420000, 0x98840000, 0x988B0000, 0x99ED0000, 0x18B80000, 0x98FA0000, 0x993A0000, 0x99A50000,
             0x19950000, 0x186D0000, 0x98330000, 0x988F0000, 0x976D0000, 0x98270000, 0x19770000, 0x99870000],
            [0x19220000, 0x99AB0000, 0x996C0000, 0x99DF0000, 0x19690000, 0x991C0000, 0x98DF0000, 0x99AB0000,
             0x18740000, 0x998C0000, 0x98AC0000, 0x98080000, 0x98290000, 0x994C0000, 0x193C0000, 0x98F30000],
            0x00000000,
        ),
        (
            "bf16",
            [0x9B940000, 0x19F30000, 0x9A2E0000, 0x1A670000, 0x9A250000, 0x9B4E0000, 0x9B290000, 0x9A330000,
             0x99E00000, 0x1A140000, 0x99140000, 0x99D10000, 0x9AE90000, 0x1B1A0000, 0x9A9C0000, 0x1B3A0000],
            [0x9AAF0000, 0x99F30000, 0x9ACC0000, 0x1B7C0000, 0x19ED0000, 0x1B250000, 0x9A1B0000, 0x9A200000,
             0x9AF70000, 0x9A480000, 0x1AE10000, 0x1B320000, 0x1B120000, 0x1B0E0000, 0x9A680000, 0x17350000],
            0x0000000B,
        ),
        (
            "tf32",
            [0x01C276FF, 0xB619A5D4, 0xA901AEAD, 0x87AC9D0C, 0x80001B54, 0x21D6A5D3, 0x36C6908E, 0x8D1E7673],
            [0x375008D5, 0x82CA0FEF, 0x8F72B73B, 0x3140C1A1, 0x170D8DC9, 0x159F25F6, 0x80B19B4F, 0x2B85B881],
            0x000001FE,
        ),
    ],
)
def test_subnormal_window_floor(a_format, a, b, expected):
    assert mma_dot(a=a, b=b, c=0, a_format=a_format) == expected
