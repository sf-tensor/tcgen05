# tcgen05-model

`tcgen05-model` is a dependency-free, bit-level software model of NVIDIA
Blackwell Tensor Core fifth-generation MMA arithmetic. It is intended for
simulation, compiler testing, and numerical analysis when executing on a GPU
is inconvenient or undesirable.

The package contains only:

- the arithmetic model and a row-major matrix interface;
- a GPU-free validation harness with frozen NVIDIA B200 result words; and
- documentation of the input formats, fixed-precision accumulator, scaling,
  sparse selection, rounding, and special values.

## Supported arithmetic

- Dense and sparse TF32, BF16, and F16
- Dense and sparse E4M3, E5M2, E2M3, E3M2, and E2M1
- Mixed f8/f6/f4 A and B formats, with F32 or F16 D
- U8/S8 integer MMA with wrapping or saturating S32 accumulation
- UE8M0 block-scaled MXF8F6F4, MXF4, and MXF4NVF4 (`scale_vec::2X/4X`)
- UE4M3-scaled NVFP4 (`scale_vec::4X`), dense and sparse

The model operates on logical matrix elements. Descriptor construction, TMEM
coordinates, and instruction scheduling are outside its scope because they do
not change the modeled arithmetic.

## Install

```bash
python -m pip install .
```

Python 3.10 or newer is required. The package has no runtime dependencies.

## Simulate an MMA

Inputs and outputs are raw container words, not Python floating-point values.
TF32 and BF16 use IEEE F32 container bits; F16 and the smaller formats use the
low bits of each integer.

```python
from tcgen05_model import f32_to_bits, bits_to_f32, mma

one = f32_to_bits(1.0)
two = f32_to_bits(2.0)
zero = f32_to_bits(0.0)

a = ((one, two),)
b = ((two,), (one,))
c = ((zero,),)

d_bits = mma(a, b, c, a_format="tf32")
assert bits_to_f32(d_bits[0][0]) == 4.0
```

For one output element, use `mma_dot`:

```python
from tcgen05_model import mma_dot

word = mma_dot(a=(0x3f800000,) * 8,
               b=(0x3f800000,) * 8,
               c=0,
               a_format="tf32")
assert word == 0x41000000  # 8.0f
```

Block scales are supplied per logical K block:

```python
word = mma_dot(
    a=(0x1,) * 64,                 # E2M1 data
    b=(0x1,) * 64,
    a_format="e2m1",
    scaling="ue4m3",
    scale_vec=4,
    scale_a=(0x38,) * 4,           # UE4M3 1.0
    scale_b=(0x38,) * 4,
)
```

Sparse calls take the instruction metadata and full logical-K vectors. An
integer metadata word is repeated for K ranges that need more than eight
nibbles; a sequence can provide distinct low/high words.

```python
word = mma_dot(
    a=(0x3f800000,) * 16,
    b=(0x3f800000,) * 16,
    a_format="tf32",
    sparse_metadata=0x44444444,
)
```

See [tcgen05_model/MODEL.md](tcgen05_model/MODEL.md) for the arithmetic rules and complete API
details.

## Validate

```bash
tcgen05-validate
# or
python -m tcgen05_model.validation
```

The validation command requires no GPU. It checks frozen B200 output words for
every unscaled floating format, all integer signedness modes, saturation,
UE8M0 MXF8F6F4/MXF4/MXF4NVF4 scaling, non-unit UE4M3 NVFP4 scaling, sparse
masking, and the matrix interface.

The CUDA harness used for the full hardware proof is also retained, but is not
installed with the package. On a machine with B200 GPUs and an `sm_100a` CUDA
toolchain, reproduce the 46-path, 46M-case validation with:

```bash
python hardware_validation/run_full_validation.py --devices 0,1
```

See [hardware_validation/README.md](hardware_validation/README.md) for hardware
requirements, reduced smoke runs, path selection, sampling details, and the
JSON result format.

For development:

```bash
python -m pip install -e '.[test]'
pytest
```

## License

MIT. See [LICENSE](LICENSE).
