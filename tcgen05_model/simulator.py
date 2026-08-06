"""Convenient scalar and row-major matrix interfaces to the tcgen05 model."""

from __future__ import annotations

from collections.abc import Sequence

from .model import DotProduct, ScalarModel, make_model, normalize_format

SPARSE_TF32_SELECTORS = frozenset((0x4, 0xE))
SPARSE_2OF4_SELECTORS = frozenset((0x4, 0x8, 0xC, 0x9, 0xD, 0x6, 0xE))


def _matrix(value: Sequence[Sequence[int]], name: str) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(row) for row in value)
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    for word in (word for row in rows for word in row):
        if not isinstance(word, int) or not 0 <= word <= 0xFFFF_FFFF:
            raise ValueError(f"{name} contains a non-uint32 word: {word!r}")
    return rows


def sparse_active_indices(
    format_name: str,
    k: int,
    metadata: int | Sequence[int],
    *,
    kind: str | None = None,
    nvfp4: bool = False,
) -> frozenset[int]:
    """Decode tcgen05 sparse-A metadata into logical K indices.

    TF32 selects one value from each pair. BF16/F16/f8/f6/f4 select two
    values from each four-wide chunk. ``kind="mxf4"``, ``"mxf4nvf4"``, or
    ``"nvfp4"`` selects two adjacent pairs from each eight-wide chunk.
    """

    format_name = normalize_format(format_name)
    kind = None if kind is None else kind.lower()
    if kind not in (None, "mxf8f6f4", "mxf4", "mxf4nvf4", "nvfp4"):
        raise ValueError(f"unsupported sparse metadata kind: {kind!r}")
    pairwise = nvfp4 or kind in ("mxf4", "mxf4nvf4", "nvfp4")
    words = (metadata,) if isinstance(metadata, int) else tuple(metadata)
    if not words or any(not 0 <= word <= 0xFFFF_FFFF for word in words):
        raise ValueError("metadata must contain uint32 words")

    if format_name == "tf32":
        chunk_size = 2
        chunks = (k + 1) // 2
        legal = SPARSE_TF32_SELECTORS
    else:
        chunk_size = 8 if pairwise else 4
        chunks = (k + chunk_size - 1) // chunk_size
        legal = SPARSE_2OF4_SELECTORS
    if isinstance(metadata, int):
        words = words * ((chunks + 7) // 8)
    if len(words) * 8 < chunks:
        raise ValueError(f"{chunks} metadata nibbles are required for K={k}")

    active: set[int] = set()
    for chunk in range(chunks):
        selector = (words[chunk // 8] >> ((chunk & 7) * 4)) & 0xF
        if selector not in legal:
            raise ValueError(f"illegal sparse selector 0x{selector:x} in chunk {chunk}")
        if format_name == "tf32":
            active.add(chunk * 2 + (selector == 0xE))
            continue
        first, second = selector & 3, (selector >> 2) & 3
        if pairwise:
            active.update((chunk * 8 + first * 2, chunk * 8 + first * 2 + 1))
            active.update((chunk * 8 + second * 2, chunk * 8 + second * 2 + 1))
        else:
            active.update((chunk * 4 + first, chunk * 4 + second))
    return frozenset(index for index in active if index < k)


def mma_dot(
    a: Sequence[int],
    b: Sequence[int],
    c: int = 0,
    *,
    model: ScalarModel | None = None,
    a_format: str = "tf32",
    b_format: str | None = None,
    d_type: str = "f32",
    saturate: bool = False,
    scaling: str | None = None,
    scale_vec: int | None = None,
    kind: str | None = None,
    scale_a: Sequence[int] | None = None,
    scale_b: Sequence[int] | None = None,
    sparse_metadata: int | Sequence[int] | None = None,
) -> int:
    """Simulate the arithmetic for one tcgen05 output element.

    All inputs and the returned value are raw container bits. For sparse MMA,
    pass the instruction metadata and full logical-K A/B vectors; disabled A
    and B positions are removed before special-value classification.
    """

    a_words, b_words = tuple(a), tuple(b)
    if len(a_words) != len(b_words):
        raise ValueError("A and B dot-product vectors must have equal length")
    sparse = sparse_metadata is not None
    if sparse:
        active = sparse_active_indices(
            a_format,
            len(a_words),
            sparse_metadata,
            kind="nvfp4" if (scaling or "").lower() == "ue4m3" else kind,
        )
        if scaling is None:
            a_words = tuple(word if index in active else 0 for index, word in enumerate(a_words))
            b_words = tuple(word if index in active else 0 for index, word in enumerate(b_words))
        else:
            logical_indices = tuple(sorted(active))
            a_words = tuple(a_words[index] for index in logical_indices)
            b_words = tuple(b_words[index] for index in logical_indices)
    selected_model = model or make_model(
        a_format,
        b_format,
        d_type=d_type,
        saturate=saturate,
        scaling=scaling,
        scale_vec=scale_vec,
        sparse=False if scaling is not None else sparse,
        kind=kind,
    )
    return selected_model.eval(
        DotProduct(
            a_words,
            b_words,
            c,
            None if scale_a is None else tuple(scale_a),
            None if scale_b is None else tuple(scale_b),
        )
    )


def mma(
    a: Sequence[Sequence[int]],
    b: Sequence[Sequence[int]],
    c: Sequence[Sequence[int]] | None = None,
    *,
    a_format: str = "tf32",
    b_format: str | None = None,
    d_type: str = "f32",
    saturate: bool = False,
    scaling: str | None = None,
    scale_vec: int | None = None,
    kind: str | None = None,
    scale_a: Sequence[Sequence[int]] | None = None,
    scale_b: Sequence[Sequence[int]] | None = None,
    sparse_metadata: int | Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Simulate ``D = A @ B + C`` on row-major raw-bit matrices.

    ``scale_a`` is indexed by A row and K block. ``scale_b`` is indexed by B
    column and K block. The result contains raw F32/F16/S32 container words.
    Shape and TMEM placement do not alter arithmetic, so this works for all
    descriptor M/N shapes supported by the instruction family.
    """

    a_rows, b_rows = _matrix(a, "A"), _matrix(b, "B")
    m, k, n = len(a_rows), len(a_rows[0]), len(b_rows[0])
    if len(b_rows) != k:
        raise ValueError(f"A is {m}x{k} but B has {len(b_rows)} rows")
    c_rows = tuple((0,) * n for _ in range(m)) if c is None else _matrix(c, "C")
    if len(c_rows) != m or len(c_rows[0]) != n:
        raise ValueError(f"C must be {m}x{n}")

    sa_rows = None if scale_a is None else _matrix(scale_a, "scale_a")
    sb_rows = None if scale_b is None else _matrix(scale_b, "scale_b")
    if (sa_rows is None) != (sb_rows is None):
        raise ValueError("scale_a and scale_b must be supplied together")
    if sa_rows is not None and (len(sa_rows) != m or len(sb_rows) != n):
        raise ValueError("scale_a needs one row per A row and scale_b one row per B column")

    model = make_model(
        a_format,
        b_format,
        d_type=d_type,
        saturate=saturate,
        scaling=scaling,
        scale_vec=scale_vec,
        sparse=sparse_metadata is not None,
        kind=kind,
    )
    result: list[tuple[int, ...]] = []
    for row in range(m):
        out_row = []
        for col in range(n):
            out_row.append(
                mma_dot(
                    a_rows[row],
                    (b_rows[index][col] for index in range(k)),
                    c_rows[row][col],
                    model=model,
                    a_format=a_format,
                    scaling=scaling,
                    kind=kind,
                    scale_a=None if sa_rows is None else sa_rows[row],
                    scale_b=None if sb_rows is None else sb_rows[col],
                    sparse_metadata=sparse_metadata,
                )
            )
        result.append(tuple(out_row))
    return tuple(result)
