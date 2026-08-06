# B200 hardware validation

This directory is the reproducible hardware side of the release validation
harness. It builds the CUDA executables that issue real `tcgen05.mma` and
`tcgen05.mma.sp` instructions, generates deterministic raw-bit inputs, and
compares every result against the model in `../tcgen05_model/`.

It is intentionally not installed by the Python wheel. The files here validate
the package; they are not required to use the simulator.

## Requirements

- NVIDIA B200 or another GPU that executes the same `sm_100a` instructions
- A CUDA toolkit with `nvcc`, `ptxas`, and `sm_100a` support
- GNU Make, Python 3.10 or newer, and a Linux host
- Substantial host memory for the 500,000-case floating-point halves. Using one
  device at a time reduces peak memory consumption.

The results in this repository were produced on B200 GPUs. Results from another
Blackwell SKU should identify that SKU in any published validation report.

## Reproduce the 46M-case run

From the repository root:

```bash
python hardware_validation/run_full_validation.py --devices 0,1
```

The driver first runs `make -j all`, then assigns one validation path at a time
to each listed GPU. With one GPU, use:

```bash
python hardware_validation/run_full_validation.py --devices 0
```

The default is 1,000,000 cases for each of 46 named harness paths, for
46,000,000 total cases. Each floating-point path is split into 500,000
unrestricted raw-encoding cases and 500,000 finite-focused cases. Integer paths
run 1,000,000 raw/adversarial cases. Random scales span their complete encoded
field, including NaN scale encodings, and sparse paths use random legal
metadata.

Progress is printed as each half completes. The process exits nonzero if a
command or worker fails, a report is malformed or incomplete, a requested path
does not complete exactly once, a model word differs, or valid output locations disagree. A
timestamped JSON report is written under `hardware_validation/results/`; it
contains every concrete command, seed, raw validator report, and failure count.
That directory is gitignored.

## Smoke tests and selecting paths

Build and exercise all paths with a small sample count:

```bash
python hardware_validation/run_full_validation.py \
  --cases-per-path 1000 --devices 0,1
```

List the 46 path names:

```bash
python hardware_validation/run_full_validation.py --list
```

Run one path or a name substring:

```bash
python hardware_validation/run_full_validation.py \
  --only dense-tf32 --cases-per-path 1000000 --devices 0

python hardware_validation/run_full_validation.py \
  --only sparse-mx --cases-per-path 10000 --devices 0
```

Use multiple `--only` arguments to select a union. Add `--skip-build` after the
runners have already been built, or build them directly with:

```bash
make -C hardware_validation -j all
```

## What is checked

The driver covers:

- dense and sparse TF32, BF16, F16, E4M3, E5M2, E2M3, E3M2, and E2M1;
- all eight U8/S8 signedness and wrapping/saturating S32 combinations;
- dense and sparse UE8M0 MXF8F6F4, MXF4, and MXF4NVF4 2X/4X paths;
- dense and sparse UE4M3 NVFP4 paths;
- directional mixed-format F32 and F16-D representatives.

Mixed format permutations instantiate one shared accumulator with independently
selected A/B decoders. The million-case matrix uses both decoder directions and
both output widths while the same-format million-case paths exercise every
individual decoder. It is representative coverage, not the complete mixed
format/shape Cartesian product.

The comparison is raw integer equality. NaNs must have the exact observed
canonical payload; numerical equivalence or NaN-class equivalence is not
accepted. Every mapped valid hardware output word must also agree with the
first word, reported as `position_failures`.

## Layout

- `run_full_validation.py`: the fixed 46-path matrix and orchestration
- `mma_probe.py`: command implementations used by the matrix
- `mma_probe/model.py`: adapter that imports the released model; it contains no
  duplicate research model
- `mma_probe/tcgen05.py`: binary case formats and CUDA process transport
- `src/*.cu`: instruction runners and Tensor Memory layouts needed for the
  validation paths
- `Makefile`: `sm_100a` builds for those runners
