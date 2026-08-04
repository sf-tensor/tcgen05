// Runtime-shape scalar harness for:
//   tcgen05.mma.cta_group::1.kind::{tf32,f16,f8f6f4,i8}
//
// Supported in this runner:
//   cta_group::1, dense and sparse A, non-.ws and .ws
//   cta_group::2, dense, non-.ws
//   cta_group::2, sparse A, non-.ws
//   cta_group::1 non-.ws M in {64, 128}, N in [8, 256] step 8
//   cta_group::1 .ws dense M in {32, 64, 128}, N in {64, 128, 256}
//   cta_group::1 .ws sparse M in {32, 64, 128}, N in {64, 128}
//   cta_group::2 non-.ws M in {128, 256}, N in [16, 256] step 16
//   cta_group::2 sparse M in {128, 256}, N in [16, 256] step 16
//   Dense K: TF32=8, BF16/F16=16, E4M3/E5M2/E2M3/E3M2/E2M1/I8=32
//   Sparse K: TF32=16, BF16/F16=32, E4M3/E5M2/E2M3/E3M2/E2M1/I8=64
//
// The fixed per-format runners remain the source of the million-case m64n8
// validations. This runner probes the descriptor shape and Tensor Memory
// datapath layout surface while reusing the same scalar dot-product setup.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kInputMagic[8] = {'M', 'M', 'A', 'P', 'R', 'B', '1', '\0'};
constexpr char kOutputMagic[8] = {'M', 'M', 'A', 'P', 'R', 'O', '1', '\0'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kOpTcgen05Shape = 28;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kCtaGroup2Threads = kThreads * 2;
constexpr uint32_t kRegsPerThread = 4;
constexpr uint32_t kTmemColumns = 512;
constexpr uint32_t kSparseMetadataOffsetCg1 = 256;
constexpr uint32_t kSparseMetadataOffsetLowCg1 = 32;
constexpr uint32_t kSparseMetadataOffsetF8F4Cg1 = 384;
constexpr uint32_t kSparseMetadataOffsetCg2 = 256;
constexpr uint32_t kWsScratchOffset = 384;
constexpr int kWsBSlotBytes = 8192;
constexpr int kMaxK = 64;
constexpr uint32_t kFlagWs = 1u;
constexpr uint32_t kFlagCtaGroup2 = 2u;
constexpr uint32_t kFlagSparse = 4u;
constexpr uint32_t kFlagSaturate = 8u;
constexpr uint32_t kFlagCoordRow = 16u;
constexpr uint32_t kFlagCoordCol = 32u;
constexpr uint32_t kFlagCoordAll = 64u;
constexpr uint32_t kFlagCoordRandom = 128u;
constexpr uint32_t kCoordFlags =
    kFlagCoordRow | kFlagCoordCol | kFlagCoordAll | kFlagCoordRandom;

enum Format : uint32_t {
  kFmtTf32 = 0,
  kFmtBf16 = 1,
  kFmtF16 = 2,
  kFmtE4m3 = 3,
  kFmtE5m2 = 4,
  kFmtE2m3 = 5,
  kFmtE3m2 = 6,
  kFmtE2m1 = 7,
  kFmtI8 = 8,
};

enum DType : uint32_t {
  kDTypeF16 = 0,
  kDTypeF32 = 1,
  kDTypeS32 = 2,
};

struct InputHeader {
  char magic[8];
  uint32_t version;
  uint32_t op;
  uint64_t count;
};

struct OutputHeader {
  char magic[8];
  uint32_t version;
  uint32_t op;
  uint32_t lanes;
  uint32_t acc_regs;
  uint64_t count;
};

struct ProbeCase {
  uint32_t format;
  uint32_t a_format;
  uint32_t b_format;
  uint32_t d_type;
  uint32_t m;
  uint32_t n;
  uint32_t flags;
  uint32_t a[kMaxK];
  uint32_t b[kMaxK];
  uint32_t c;
  uint32_t metadata;
  uint32_t metadata_hi;
};

struct alignas(128) SharedStorage {
  alignas(16) uint32_t tmem_base;
  alignas(8) uint64_t mma_barrier;
  alignas(8) uint64_t collector_barrier;
  alignas(128) uint8_t a_smem[8192];
  alignas(128) uint8_t b_smem[kWsBSlotBytes * 4];
};

static_assert(sizeof(InputHeader) == 24, "unexpected InputHeader packing");
static_assert(sizeof(OutputHeader) == 32, "unexpected OutputHeader packing");
static_assert(sizeof(ProbeCase) == 552, "unexpected ProbeCase packing");

void die_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::string msg = std::string(what) + ": " + cudaGetErrorString(err);
    throw std::runtime_error(msg);
  }
}

template <class T>
void read_exact(std::istream& in, T* data, size_t count, const char* what) {
  const auto bytes = static_cast<std::streamsize>(sizeof(T) * count);
  in.read(reinterpret_cast<char*>(data), bytes);
  if (!in || in.gcount() != bytes) {
    throw std::runtime_error(std::string("short read while reading ") + what);
  }
}

template <class T>
void write_exact(std::ostream& out, const T* data, size_t count, const char* what) {
  const auto bytes = static_cast<std::streamsize>(sizeof(T) * count);
  out.write(reinterpret_cast<const char*>(data), bytes);
  if (!out) {
    throw std::runtime_error(std::string("short write while writing ") + what);
  }
}

bool valid_format(uint32_t format) {
  return format <= kFmtI8;
}

__host__ __device__ __forceinline__ bool is_f8f6f4_format(uint32_t format) {
  return format >= kFmtE4m3 && format <= kFmtE2m1;
}

bool valid_dtype(uint32_t d_type) {
  return d_type == kDTypeF16 || d_type == kDTypeF32 || d_type == kDTypeS32;
}

bool valid_type_combo(uint32_t format, uint32_t a_format, uint32_t b_format, uint32_t d_type, uint32_t flags) {
  (void)flags;
  if (!valid_format(format) || !valid_dtype(d_type)) {
    return false;
  }
  if (format == kFmtI8) {
    return d_type == kDTypeS32 && a_format <= 1u && b_format <= 1u;
  }
  if (!valid_format(a_format) || !valid_format(b_format) || d_type == kDTypeS32) {
    return false;
  }
  if (format == kFmtTf32) {
    return a_format == kFmtTf32 && b_format == kFmtTf32 && d_type == kDTypeF32;
  }
  if (format == kFmtBf16 || format == kFmtF16) {
    if ((a_format != kFmtBf16 && a_format != kFmtF16) || a_format != b_format) {
      return false;
    }
    return d_type == kDTypeF32 || a_format == kFmtF16;
  }
  if (is_f8f6f4_format(format)) {
    return is_f8f6f4_format(a_format) && is_f8f6f4_format(b_format);
  }
  return false;
}

bool valid_i8_n(uint32_t n) {
  return n == 8 || n == 16 || n == 24 || n == 32 ||
         (n >= 48 && n <= 256 && (n % 16) == 0);
}

bool valid_shape(uint32_t format, uint32_t m, uint32_t n, uint32_t flags) {
  const bool ws = (flags & kFlagWs) != 0;
  const bool cta_group2 = (flags & kFlagCtaGroup2) != 0;
  const bool sparse = (flags & kFlagSparse) != 0;
  if ((flags & ~(kFlagWs | kFlagCtaGroup2 | kFlagSparse | kFlagSaturate | kCoordFlags)) != 0) {
    return false;
  }
  const uint32_t coord_flags = flags & kCoordFlags;
  if (coord_flags != 0 && coord_flags != kFlagCoordRow &&
      coord_flags != kFlagCoordCol && coord_flags != kFlagCoordAll &&
      coord_flags != kFlagCoordRandom) {
    return false;
  }
  if (format != kFmtI8 && (flags & kFlagSaturate) != 0) {
    return false;
  }
  if (format == kFmtI8) {
    if (sparse) {
      if (cta_group2) {
        return !ws && (m == 128 || m == 256) && n >= 32 && n <= 256 && (n % 32) == 0;
      }
      if (ws) {
        return (m == 32 || m == 64 || m == 128) && (n == 64 || n == 128);
      }
      return (m == 64 || m == 128) && valid_i8_n(n);
    }
    if (cta_group2) {
      return !ws && (m == 128 || m == 256) && n >= 32 && n <= 256 && (n % 32) == 0;
    }
    if (ws) {
      return (m == 32 || m == 64 || m == 128) && (n == 64 || n == 128 || n == 256);
    }
    return (m == 64 || m == 128) && valid_i8_n(n);
  }
  if (sparse) {
    if (cta_group2) {
      return !ws && (m == 128 || m == 256) && n >= 16 && n <= 256 && (n % 16) == 0;
    }
    if (ws) {
      return (m == 32 || m == 64 || m == 128) && (n == 64 || n == 128);
    }
    return (m == 64 || m == 128) && n >= 8 && n <= 256 && (n % 8) == 0;
  }
  if (cta_group2) {
    return !ws && (m == 128 || m == 256) && n >= 16 && n <= 256 && (n % 16) == 0;
  }
  if (ws) {
    return (m == 32 || m == 64 || m == 128) && (n == 64 || n == 128 || n == 256);
  }
  return (m == 64 || m == 128) && n >= 8 && n <= 256 && (n % 8) == 0;
}

bool valid_sparse_metadata_word(uint32_t metadata, bool tf32) {
  for (int chunk = 0; chunk < 8; ++chunk) {
    const uint32_t selector = (metadata >> (chunk * 4)) & 0xFu;
    if (tf32) {
      if (selector != 0x4u && selector != 0xEu) {
        return false;
      }
    } else if (selector != 0x4u && selector != 0x8u && selector != 0xCu &&
               selector != 0x9u && selector != 0xDu && selector != 0x6u &&
               selector != 0xEu) {
      return false;
    }
  }
  return true;
}

bool valid_sparse_metadata(const ProbeCase& pc) {
  if ((pc.flags & kFlagSparse) == 0) {
    return true;
  }
  const bool tf32 = pc.format == kFmtTf32;
  return valid_sparse_metadata_word(pc.metadata, tf32) &&
         valid_sparse_metadata_word(pc.metadata_hi, tf32);
}

std::vector<ProbeCase> read_cases(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open input file: " + path);
  }

  InputHeader header{};
  read_exact(in, &header, 1, "input header");
  if (std::memcmp(header.magic, kInputMagic, sizeof(header.magic)) != 0) {
    throw std::runtime_error("bad input magic");
  }
  if (header.version != kVersion) {
    throw std::runtime_error("unsupported input version");
  }
  if (header.op != kOpTcgen05Shape) {
    throw std::runtime_error("unsupported op id");
  }
  if (header.count > (1ull << 34)) {
    throw std::runtime_error("case count is unreasonable");
  }

  std::vector<ProbeCase> cases(static_cast<size_t>(header.count));
  if (!cases.empty()) {
    read_exact(in, cases.data(), cases.size(), "cases");
  }
  for (const ProbeCase& pc : cases) {
    if (!valid_type_combo(pc.format, pc.a_format, pc.b_format, pc.d_type, pc.flags) ||
        !valid_shape(pc.format, pc.m, pc.n, pc.flags) ||
        !valid_sparse_metadata(pc)) {
      throw std::runtime_error("unsupported format or shape in input case");
    }
  }
  return cases;
}

void write_outputs(const std::string& path,
                   const std::vector<uint32_t>& outputs,
                   size_t count,
                   uint32_t threads_per_case) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }

  OutputHeader header{};
  std::memcpy(header.magic, kOutputMagic, sizeof(header.magic));
  header.version = kVersion;
  header.op = kOpTcgen05Shape;
  header.lanes = threads_per_case;
  header.acc_regs = kRegsPerThread;
  header.count = count;

  write_exact(out, &header, 1, "output header");
  if (!outputs.empty()) {
    write_exact(out, outputs.data(), outputs.size(), "outputs");
  }
}

uint32_t output_threads_per_case(const std::vector<ProbeCase>& cases) {
  if (cases.empty()) {
    return kThreads;
  }
  const bool cta_group2 = (cases.front().flags & kFlagCtaGroup2) != 0;
  const bool coordinate_mode = (cases.front().flags & kCoordFlags) != 0;
  const uint32_t base_threads_per_case = cta_group2 ? kCtaGroup2Threads : kThreads;
  if (!coordinate_mode) {
    return base_threads_per_case;
  }
  const bool ws = (cases.front().flags & kFlagWs) != 0;
  const uint32_t row_groups =
      cta_group2 || ws ? 2u : (cases.front().m <= 64u ? 1u : cases.front().m / 64u);
  const uint32_t col_divisor = (ws && cases.front().m == 64u) ? 16u : 8u;
  const uint32_t raw_col_groups = cases.front().n / col_divisor;
  const uint32_t col_groups = raw_col_groups == 0 ? 1u : raw_col_groups;
  return base_threads_per_case * row_groups * col_groups;
}

__host__ __device__ __forceinline__ uint32_t f8f6f4_type_code(uint32_t format) {
  switch (format) {
    case kFmtE4m3:
      return 0;
    case kFmtE5m2:
      return 1;
    case kFmtE2m3:
      return 3;
    case kFmtE3m2:
      return 4;
    case kFmtE2m1:
      return 5;
    default:
      return 0;
  }
}

__host__ __device__ __forceinline__ uint32_t f16_type_code(uint32_t format) {
  return format == kFmtBf16 ? 1u : 0u;
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr,
                                                   uint32_t leading_bytes,
                                                   uint32_t stride_bytes) {
  const uint64_t start = static_cast<uint64_t>(smem_u32(ptr) >> 4);
  const uint64_t leading = static_cast<uint64_t>((leading_bytes & 0x3ffffu) >> 4);
  const uint64_t stride = static_cast<uint64_t>((stride_bytes & 0x3ffffu) >> 4);
  uint64_t desc = 0;
  desc |= start;
  desc |= leading << 16;
  desc |= stride << 32;
  desc |= 1ull << 46;
  return desc;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t format,
                                                    uint32_t a_format,
                                                    uint32_t b_format,
                                                    uint32_t d_type,
                                                    uint32_t m,
                                                    uint32_t n) {
  uint32_t desc = 0;
  desc |= d_type << 4;
  if (format == kFmtTf32) {
    desc |= 2u << 7;
    desc |= 2u << 10;
  } else if (format == kFmtBf16 || format == kFmtF16) {
    desc |= f16_type_code(a_format) << 7;
    desc |= f16_type_code(b_format) << 10;
  } else {
    desc |= f8f6f4_type_code(a_format) << 7;
    desc |= f8f6f4_type_code(b_format) << 10;
  }
  desc |= (n >> 3) << 17;
  desc |= (m >> 4) << 24;
  return desc;
}

__device__ __forceinline__ uint32_t make_f8f6f4_instr_desc_for_types(uint32_t a_format,
                                                                     uint32_t b_format,
                                                                     uint32_t d_type,
                                                                     uint32_t m,
                                                                     uint32_t n) {
  return make_instr_desc(kFmtE4m3, a_format, b_format, d_type, m, n);
}

__device__ __forceinline__ uint32_t make_i8_instr_desc(uint32_t a_type,
                                                       uint32_t b_type,
                                                       uint32_t flags,
                                                       uint32_t m,
                                                       uint32_t n) {
  uint32_t desc = 0;
  desc |= ((flags & kFlagSaturate) ? 1u : 0u) << 3;
  desc |= kDTypeS32 << 4;
  desc |= (a_type & 1u) << 7;
  desc |= (b_type & 1u) << 10;
  desc |= (n >> 3) << 17;
  desc |= (m >> 4) << 24;
  return desc;
}

__device__ __forceinline__ uint32_t make_ws_instr_desc(uint32_t format,
                                                       uint32_t a_format,
                                                       uint32_t b_format,
                                                       uint32_t d_type,
                                                       uint32_t m,
                                                       uint32_t n) {
  // M32 .ws traps if the B-reuse shift permits movement outside the 32-row tile.
  const uint32_t max_shift = (m == 32) ? 0u : 3u;
  return make_instr_desc(format, a_format, b_format, d_type, m, n) | (max_shift << 30);
}

__device__ __forceinline__ uint32_t make_f8f6f4_ws_instr_desc(uint32_t a_format,
                                                              uint32_t b_format,
                                                              uint32_t d_type,
                                                              uint32_t m,
                                                              uint32_t n) {
  // M32 .ws traps if the B-reuse shift permits movement outside the 32-row tile.
  const uint32_t max_shift = (m == 32) ? 0u : 3u;
  return make_f8f6f4_instr_desc_for_types(a_format, b_format, d_type, m, n) |
         (max_shift << 30);
}

__device__ __forceinline__ uint32_t make_i8_ws_instr_desc(uint32_t a_type,
                                                          uint32_t b_type,
                                                          uint32_t flags,
                                                          uint32_t m,
                                                          uint32_t n) {
  const uint32_t max_shift = (m == 32) ? 0u : 3u;
  return make_i8_instr_desc(a_type, b_type, flags, m, n) | (max_shift << 30);
}

__device__ __forceinline__ int k_major_offset(int row_or_col, int k, int t) {
  return (row_or_col & 7) * t + (row_or_col >> 3) * 32 + (k & (t - 1)) + (k / t) * 64;
}

__device__ __forceinline__ int coordinate_lbo_units(int t) {
  return t == 16 ? 128 : 64;
}

__device__ __forceinline__ int coordinate_sbo_units(int t, int storage_k) {
  const int k_blocks = (storage_k + t - 1) / t;
  return coordinate_lbo_units(t) * k_blocks;
}

__device__ __forceinline__ int k_major_offset_for(const ProbeCase* pc,
                                                  int row_or_col,
                                                  int k,
                                                  int t,
                                                  int storage_k) {
  if ((pc->flags & kCoordFlags) == 0) {
    return k_major_offset(row_or_col, k, t);
  }
  const int lbo = coordinate_lbo_units(t);
  const int sbo = coordinate_sbo_units(t, storage_k);
  return (row_or_col & 7) * t + (row_or_col >> 3) * sbo + (k & (t - 1)) +
         (k / t) * lbo;
}

__device__ __forceinline__ int f8f6f4_storage_k(uint32_t format) {
  const uint32_t type_code = f8f6f4_type_code(format);
  if (type_code == 3 || type_code == 4) {
    return 24;
  }
  if (type_code == 5) {
    return 16;
  }
  return 32;
}

__device__ __forceinline__ int f8f6f4_coordinate_storage_k(uint32_t format) {
  const uint32_t type_code = f8f6f4_type_code(format);
  if (type_code == 5) {
    return 32;
  }
  return f8f6f4_storage_k(format);
}

__device__ __forceinline__ int operand_coordinate_storage_k(uint32_t format,
                                                            uint32_t operand_format,
                                                            bool sparse,
                                                            bool matrix_a) {
  if (format == kFmtTf32) {
    return sparse && !matrix_a ? 16 : 8;
  }
  if (format == kFmtBf16 || format == kFmtF16) {
    return sparse && !matrix_a ? 32 : 16;
  }
  if (format == kFmtI8) {
    return sparse && !matrix_a ? 64 : 32;
  }
  if (sparse && !matrix_a) {
    return 64;
  }
  return f8f6f4_coordinate_storage_k(operand_format);
}

__device__ __forceinline__ uint32_t coordinate_lbo_bytes(uint32_t format, uint32_t operand_format) {
  if (format == kFmtTf32) {
    return static_cast<uint32_t>(coordinate_lbo_units(4) * sizeof(uint32_t));
  }
  if (format == kFmtBf16 || format == kFmtF16) {
    return static_cast<uint32_t>(coordinate_lbo_units(8) * sizeof(uint16_t));
  }
  (void)operand_format;
  return static_cast<uint32_t>(coordinate_lbo_units(16) * sizeof(uint8_t));
}

__device__ __forceinline__ uint32_t coordinate_sbo_bytes(uint32_t format,
                                                         uint32_t operand_format,
                                                         bool sparse,
                                                         bool matrix_a) {
  if (format == kFmtTf32) {
    const int storage_k = operand_coordinate_storage_k(format, operand_format, sparse, matrix_a);
    return static_cast<uint32_t>(coordinate_sbo_units(4, storage_k) * sizeof(uint32_t));
  }
  if (format == kFmtBf16 || format == kFmtF16) {
    const int storage_k = operand_coordinate_storage_k(format, operand_format, sparse, matrix_a);
    return static_cast<uint32_t>(coordinate_sbo_units(8, storage_k) * sizeof(uint16_t));
  }
  const int storage_k = operand_coordinate_storage_k(format, operand_format, sparse, matrix_a);
  return static_cast<uint32_t>(coordinate_sbo_units(16, storage_k) * sizeof(uint8_t));
}

__device__ __forceinline__ uint32_t operand_lbo_bytes(uint32_t format,
                                                      uint32_t operand_format,
                                                      bool coordinate_mode) {
  if (coordinate_mode) {
    return coordinate_lbo_bytes(format, operand_format);
  }
  if (format == kFmtTf32) {
    return 64 * sizeof(uint32_t);
  }
  if (format == kFmtBf16 || format == kFmtF16) {
    return 64 * sizeof(uint16_t);
  }
  return 64 * sizeof(uint8_t);
}

__device__ __forceinline__ uint32_t operand_sbo_bytes(uint32_t format,
                                                      uint32_t operand_format,
                                                      bool coordinate_mode,
                                                      bool sparse,
                                                      bool matrix_a) {
  if (coordinate_mode) {
    return coordinate_sbo_bytes(format, operand_format, sparse, matrix_a);
  }
  if (format == kFmtTf32) {
    return 32 * sizeof(uint32_t);
  }
  if (format == kFmtBf16 || format == kFmtF16) {
    return 32 * sizeof(uint16_t);
  }
  return 32 * sizeof(uint8_t);
}

__host__ __device__ __forceinline__ uint32_t shape_one_word(uint32_t format) {
  switch (format) {
    case kFmtTf32:
    case kFmtBf16:
      return 0x3f800000u;
    case kFmtF16:
      return 0x3c00u;
    case kFmtE4m3:
      return 0x38u;
    case kFmtE5m2:
      return 0x3cu;
    case kFmtE2m3:
      return 0x08u;
    case kFmtE3m2:
      return 0x0cu;
    case kFmtE2m1:
      return 0x02u;
    case kFmtI8:
      return 1u;
    default:
      return 0u;
  }
}

__device__ __forceinline__ uint32_t operand_one_word(const ProbeCase* pc, uint32_t operand_format) {
  if (pc->format == kFmtI8) {
    return 1u;
  }
  return shape_one_word(operand_format);
}

__host__ __device__ __forceinline__ uint32_t coord_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ uint32_t coord_hash(uint32_t seed,
                                                        uint32_t a,
                                                        uint32_t b,
                                                        uint32_t c,
                                                        uint32_t tag) {
  uint32_t x = seed ^ 0x9e3779b9u;
  x ^= a * 0x85ebca6bu;
  x ^= b * 0xc2b2ae35u;
  x ^= c * 0x27d4eb2du;
  x ^= tag * 0x165667b1u;
  return coord_mix32(x);
}

__device__ __forceinline__ uint32_t coordinate_random_f32_word(uint32_t h) {
  switch (h % 6u) {
    case 0:
      return 0x3f800000u;  // 1.0
    case 1:
      return 0xbf800000u;  // -1.0
    case 2:
      return 0x40000000u;  // 2.0
    case 3:
      return 0xc0000000u;  // -2.0
    case 4:
      return 0x3f000000u;  // 0.5
    default:
      return 0xbf000000u;  // -0.5
  }
}

__device__ __forceinline__ uint32_t coordinate_random_f16_word(uint32_t h) {
  switch (h % 6u) {
    case 0:
      return 0x3c00u;  // 1.0
    case 1:
      return 0xbc00u;  // -1.0
    case 2:
      return 0x4000u;  // 2.0
    case 3:
      return 0xc000u;  // -2.0
    case 4:
      return 0x3800u;  // 0.5
    default:
      return 0xb800u;  // -0.5
  }
}

__device__ __forceinline__ uint32_t coordinate_random_i8_word(uint32_t h,
                                                              uint32_t operand_format) {
  if (operand_format == 1u) {
    switch (h % 7u) {
      case 0:
        return static_cast<uint32_t>(static_cast<uint8_t>(-3));
      case 1:
        return static_cast<uint32_t>(static_cast<uint8_t>(-2));
      case 2:
        return static_cast<uint32_t>(static_cast<uint8_t>(-1));
      case 3:
        return 1u;
      case 4:
        return 2u;
      case 5:
        return 3u;
      default:
        return 0u;
    }
  }
  return (h % 7u) + 1u;
}

__device__ __forceinline__ uint32_t coordinate_random_f8f6f4_word(uint32_t format,
                                                                  uint32_t h) {
  const uint32_t slot = h % 6u;
  if (format == kFmtE5m2) {
    switch (slot) {
      case 0:
        return 0x3cu;
      case 1:
        return 0xbcu;
      case 2:
        return 0x40u;
      case 3:
        return 0xc0u;
      case 4:
        return 0x38u;
      default:
        return 0xb8u;
    }
  }
  if (format == kFmtE2m3) {
    switch (slot) {
      case 0:
        return 0x08u;
      case 1:
        return 0x28u;
      case 2:
        return 0x10u;
      case 3:
        return 0x30u;
      case 4:
        return 0x04u;
      default:
        return 0x24u;
    }
  }
  if (format == kFmtE3m2) {
    switch (slot) {
      case 0:
        return 0x0cu;
      case 1:
        return 0x2cu;
      case 2:
        return 0x10u;
      case 3:
        return 0x30u;
      case 4:
        return 0x08u;
      default:
        return 0x28u;
    }
  }
  if (format == kFmtE2m1) {
    switch (slot) {
      case 0:
        return 0x02u;
      case 1:
        return 0x0au;
      case 2:
        return 0x04u;
      case 3:
        return 0x0cu;
      case 4:
        return 0x01u;
      default:
        return 0x09u;
    }
  }
  switch (slot) {
    case 0:
      return 0x38u;
    case 1:
      return 0xb8u;
    case 2:
      return 0x40u;
    case 3:
      return 0xc0u;
    case 4:
      return 0x30u;
    default:
      return 0xb0u;
  }
}

__device__ __forceinline__ uint32_t coordinate_random_operand_word(const ProbeCase* pc,
                                                                   uint32_t operand_format,
                                                                   int row_or_col,
                                                                   int k,
                                                                   bool matrix_a) {
  const uint32_t tag = matrix_a ? 0xa341316cu : 0xc8013ea4u;
  const uint32_t h = coord_hash(pc->c,
                                static_cast<uint32_t>(row_or_col),
                                static_cast<uint32_t>(k),
                                operand_format,
                                tag);
  if (pc->format == kFmtI8) {
    return coordinate_random_i8_word(h, operand_format);
  }
  if (operand_format == kFmtTf32 || operand_format == kFmtBf16) {
    return coordinate_random_f32_word(h);
  }
  if (operand_format == kFmtF16) {
    return coordinate_random_f16_word(h);
  }
  return coordinate_random_f8f6f4_word(operand_format, h);
}

__device__ __forceinline__ uint32_t coordinate_random_c_word(const ProbeCase* pc,
                                                            uint32_t d_type,
                                                            uint32_t group,
                                                            uint32_t thread,
                                                            uint32_t reg) {
  const uint32_t h = coord_hash(pc->c, group, thread, reg, 0xd1b54a32u);
  if (d_type == kDTypeF16) {
    return coordinate_random_f16_word(h);
  }
  if (d_type == kDTypeS32) {
    switch (h % 9u) {
      case 0:
        return static_cast<uint32_t>(-17);
      case 1:
        return static_cast<uint32_t>(-9);
      case 2:
        return static_cast<uint32_t>(-3);
      case 3:
        return 0u;
      case 4:
        return 5u;
      case 5:
        return 11u;
      case 6:
        return 19u;
      case 7:
        return static_cast<uint32_t>(-23);
      default:
        return 29u;
    }
  }
  return coordinate_random_f32_word(h);
}

__device__ __forceinline__ uint32_t operand_word(const ProbeCase* pc,
                                                 const uint32_t* values,
                                                 uint32_t operand_format,
                                                 int row_or_col,
                                                 int k,
                                                 bool matrix_a) {
  const uint32_t coord_flags = pc->flags & kCoordFlags;
  if (coord_flags == 0) {
    return values[k];
  }
  if ((coord_flags & kFlagCoordRandom) != 0) {
    return coordinate_random_operand_word(pc, operand_format, row_or_col, k, matrix_a);
  }
  const uint32_t one = operand_one_word(pc, operand_format);
  if ((coord_flags & kFlagCoordAll) != 0) {
    return one;
  }
  const uint32_t bit = pc->c & 31u;
  const bool selected = ((static_cast<uint32_t>(row_or_col) >> bit) & 1u) != 0;
  if ((coord_flags & kFlagCoordRow) != 0) {
    return matrix_a ? (selected ? one : 0u) : one;
  }
  return matrix_a ? one : (selected ? one : 0u);
}

__device__ __forceinline__ uint8_t pack_f6_byte0(uint32_t v0, uint32_t v1) {
  return static_cast<uint8_t>((v0 & 0x3fu) | ((v1 & 0x3u) << 6));
}

__device__ __forceinline__ uint8_t pack_f6_byte1(uint32_t v1, uint32_t v2) {
  return static_cast<uint8_t>(((v1 >> 2) & 0xFu) | ((v2 & 0xFu) << 4));
}

__device__ __forceinline__ uint8_t pack_f6_byte2(uint32_t v2, uint32_t v3) {
  return static_cast<uint8_t>(((v2 >> 4) & 0x3u) | ((v3 & 0x3Fu) << 2));
}

__device__ __forceinline__ uint8_t pack_f4_pair(uint32_t v0, uint32_t v1) {
  return static_cast<uint8_t>((v0 & 0xFu) | ((v1 & 0xFu) << 4));
}

__device__ __forceinline__ void tmem_store_16x256(uint32_t taddr, uint32_t value) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %1, %1, %1};\n"
      :
      : "r"(taddr), "r"(value)
      : "memory");
}

__device__ __forceinline__ void tmem_store_16x256_words(uint32_t taddr,
                                                       uint32_t r0,
                                                       uint32_t r1,
                                                       uint32_t r2,
                                                       uint32_t r3) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %2, %3, %4};\n"
      :
      : "r"(taddr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
      : "memory");
}

__device__ __forceinline__ void tmem_store_sparse_metadata_short(uint32_t taddr,
                                                                uint32_t metadata) {
  const uint32_t lo = metadata & 0xFFFFu;
  const uint32_t hi = (metadata >> 16) & 0xFFFFu;
  const uint32_t lo_word = lo | (lo << 16);
  const uint32_t hi_word = hi | (hi << 16);
  tmem_store_16x256_words(taddr, lo_word, lo_word, hi_word, hi_word);
}

__device__ __forceinline__ void tmem_store_sparse_metadata(uint32_t taddr,
                                                          uint32_t format,
                                                          uint32_t metadata,
                                                          uint32_t metadata_hi) {
  if (format == kFmtTf32 || format == kFmtBf16 || format == kFmtF16) {
    tmem_store_sparse_metadata_short(taddr, metadata);
  } else {
    tmem_store_16x256_words(taddr, metadata, metadata_hi, metadata, metadata_hi);
  }
}

__device__ __forceinline__ uint32_t c_store_word(uint32_t c, uint32_t d_type) {
  if (d_type == kDTypeF16) {
    const uint32_t half = c & 0xFFFFu;
    return half | (half << 16);
  }
  return c;
}

__device__ __forceinline__ void tmem_load_16x256(uint32_t taddr,
                                                 uint32_t& r0,
                                                 uint32_t& r1,
                                                 uint32_t& r2,
                                                 uint32_t& r3) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(taddr)
      : "memory");
}

__device__ __forceinline__ uint32_t tmem_addr_lane(uint32_t tbase, uint32_t lane) {
  return tbase | (lane << 16);
}

__host__ __device__ __forceinline__ uint32_t coordinate_row_groups(uint32_t m, uint32_t flags) {
  if ((flags & kFlagCtaGroup2) != 0 || (flags & kFlagWs) != 0) {
    return 2u;
  }
  return m <= 64u ? 1u : (m / 64u);
}

__host__ __device__ __forceinline__ uint32_t coordinate_cta_row_base(uint32_t m,
                                                                     uint32_t flags,
                                                                     uint32_t pair_rank) {
  if ((flags & kFlagCtaGroup2) != 0) {
    return pair_rank * (m / 2u);
  }
  return 0u;
}

__host__ __device__ __forceinline__ uint32_t coordinate_col_groups(uint32_t m,
                                                                   uint32_t n,
                                                                   uint32_t flags) {
  const bool ws = (flags & kFlagWs) != 0;
  const uint32_t col_divisor = (ws && m == 64u) ? 16u : 8u;
  const uint32_t raw = n / col_divisor;
  return raw == 0 ? 1u : raw;
}

__host__ __device__ __forceinline__ uint32_t coordinate_load_groups(uint32_t m,
                                                                    uint32_t n,
                                                                    uint32_t flags) {
  if ((flags & kCoordFlags) == 0) {
    return 1u;
  }
  return coordinate_row_groups(m, flags) * coordinate_col_groups(m, n, flags);
}

__device__ __forceinline__ uint32_t coordinate_tmem_addr(uint32_t tbase,
                                                        int warp,
                                                        uint32_t group,
                                                        uint32_t m,
                                                        uint32_t n,
                                                        uint32_t flags) {
  const uint32_t row_groups = coordinate_row_groups(m, flags);
  const uint32_t row_group = group % row_groups;
  const uint32_t col_group = group / row_groups;
  const uint32_t lane = static_cast<uint32_t>(warp * 32) + row_group * 16u;
  const uint32_t column = col_group * 8u;
  (void)n;
  return tmem_addr_lane(tbase + column, lane);
}

__device__ __forceinline__ uint32_t coordinate_b_column_byte_offset(const ProbeCase* pc,
                                                                    uint32_t format,
                                                                    uint32_t b_format,
                                                                    bool sparse,
                                                                    uint32_t col) {
  const int storage_k = operand_coordinate_storage_k(format, b_format, sparse, false);
  if (format == kFmtTf32) {
    return static_cast<uint32_t>(k_major_offset_for(pc, static_cast<int>(col), 0, 4, storage_k) *
                                 static_cast<int>(sizeof(uint32_t)));
  }
  if (format == kFmtBf16 || format == kFmtF16) {
    return static_cast<uint32_t>(k_major_offset_for(pc, static_cast<int>(col), 0, 8, storage_k) *
                                 static_cast<int>(sizeof(uint16_t)));
  }
  return static_cast<uint32_t>(k_major_offset_for(pc, static_cast<int>(col), 0, 16, storage_k));
}

__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void proxy_async_fence_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile(
      "barrier.cluster.arrive;\n\t"
      "barrier.cluster.wait;\n"
      :
      :
      : "memory");
}

__device__ __forceinline__ void init_barrier(uint64_t* barrier, uint32_t arrivals) {
  const uint32_t addr = smem_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(addr), "r"(arrivals) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* barrier, uint32_t phase) {
  const uint32_t addr = smem_u32(barrier);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "wait_loop:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra wait_loop;\n\t"
      "}\n"
      :
      : "r"(addr), "r"(phase)
      : "memory");
}

__device__ __forceinline__ void tcgen05_commit(uint64_t* barrier) {
  const uint32_t addr = smem_u32(barrier);
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      :
      : "r"(addr)
      : "memory");
}

__device__ __forceinline__ void tcgen05_commit_cg2(uint64_t* barrier) {
  const uint32_t addr = smem_u32(barrier);
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      :
      : "r"(addr)
      : "memory");
}

__device__ __forceinline__ void issue_tf32(uint32_t tmem_base,
                                          uint64_t desc_a,
                                          uint64_t desc_b,
                                          uint32_t idesc,
                                          uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_f16(uint32_t tmem_base,
                                         uint64_t desc_a,
                                         uint64_t desc_b,
                                         uint32_t idesc,
                                         uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_f8f6f4(uint32_t tmem_base,
                                            uint64_t desc_a,
                                            uint64_t desc_b,
                                            uint32_t idesc,
                                            uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_i8(uint32_t tmem_base,
                                         uint64_t desc_a,
                                         uint64_t desc_b,
                                         uint32_t idesc,
                                         uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_tf32_cg2(uint32_t tmem_base,
                                               uint64_t desc_a,
                                               uint64_t desc_b,
                                               uint32_t idesc,
                                               uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, "
      "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3), "r"(mask4), "r"(mask5),
        "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_f16_cg2(uint32_t tmem_base,
                                              uint64_t desc_a,
                                              uint64_t desc_b,
                                              uint32_t idesc,
                                              uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, "
      "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3), "r"(mask4), "r"(mask5),
        "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_f8f6f4_cg2(uint32_t tmem_base,
                                                 uint64_t desc_a,
                                                 uint64_t desc_b,
                                                 uint32_t idesc,
                                                 uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, "
      "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3), "r"(mask4), "r"(mask5),
        "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_i8_cg2(uint32_t tmem_base,
                                             uint64_t desc_a,
                                             uint64_t desc_b,
                                             uint32_t idesc,
                                             uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::i8 [%0], %1, %2, %3, "
      "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
        "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3), "r"(mask4), "r"(mask5),
        "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_tf32_sparse_cg2(uint32_t tmem_base,
                                                      uint64_t desc_a,
                                                      uint64_t desc_b,
                                                      uint32_t sp_meta_tmem,
                                                      uint32_t idesc,
                                                      uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::2.kind::tf32 [%0], %1, %2, [%3], %4, "
      "{%6, %7, %8, %9, %10, %11, %12, %13}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3),
        "r"(mask4), "r"(mask5), "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_f16_sparse_cg2(uint32_t tmem_base,
                                                     uint64_t desc_a,
                                                     uint64_t desc_b,
                                                     uint32_t sp_meta_tmem,
                                                     uint32_t idesc,
                                                     uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::2.kind::f16 [%0], %1, %2, [%3], %4, "
      "{%6, %7, %8, %9, %10, %11, %12, %13}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3),
        "r"(mask4), "r"(mask5), "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_f8f6f4_sparse_cg2(uint32_t tmem_base,
                                                        uint64_t desc_a,
                                                        uint64_t desc_b,
                                                        uint32_t sp_meta_tmem,
                                                        uint32_t idesc,
                                                        uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [%0], %1, %2, [%3], %4, "
      "{%6, %7, %8, %9, %10, %11, %12, %13}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3),
        "r"(mask4), "r"(mask5), "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_i8_sparse_cg2(uint32_t tmem_base,
                                                    uint64_t desc_a,
                                                    uint64_t desc_b,
                                                    uint32_t sp_meta_tmem,
                                                    uint32_t idesc,
                                                    uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  uint32_t mask4 = 0;
  uint32_t mask5 = 0;
  uint32_t mask6 = 0;
  uint32_t mask7 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::2.kind::i8 [%0], %1, %2, [%3], %4, "
      "{%6, %7, %8, %9, %10, %11, %12, %13}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3),
        "r"(mask4), "r"(mask5), "r"(mask6), "r"(mask7)
      : "memory");
}

__device__ __forceinline__ void issue_tf32_sparse(uint32_t tmem_base,
                                                  uint64_t desc_a,
                                                  uint64_t desc_b,
                                                  uint32_t sp_meta_tmem,
                                                  uint32_t idesc,
                                                  uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::1.kind::tf32 [%0], %1, %2, [%3], %4, {%6, %7, %8, %9}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_f16_sparse(uint32_t tmem_base,
                                                 uint64_t desc_a,
                                                 uint64_t desc_b,
                                                 uint32_t sp_meta_tmem,
                                                 uint32_t idesc,
                                                 uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%3], %4, {%6, %7, %8, %9}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_f8f6f4_sparse(uint32_t tmem_base,
                                                    uint64_t desc_a,
                                                    uint64_t desc_b,
                                                    uint32_t sp_meta_tmem,
                                                    uint32_t idesc,
                                                    uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [%0], %1, %2, [%3], %4, {%6, %7, %8, %9}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_i8_sparse(uint32_t tmem_base,
                                                uint64_t desc_a,
                                                uint64_t desc_b,
                                                uint32_t sp_meta_tmem,
                                                uint32_t idesc,
                                                uint32_t enable_input_d) {
  uint32_t mask0 = 0;
  uint32_t mask1 = 0;
  uint32_t mask2 = 0;
  uint32_t mask3 = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "tcgen05.mma.sp.cta_group::1.kind::i8 [%0], %1, %2, [%3], %4, {%6, %7, %8, %9}, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),
        "r"(enable_input_d), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}

__device__ __forceinline__ void issue_tf32_ws(uint32_t tmem_base,
                                             uint64_t desc_a,
                                             uint64_t desc_b,
                                             uint32_t idesc,
                                             uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::discard [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_f16_ws(uint32_t tmem_base,
                                            uint64_t desc_a,
                                            uint64_t desc_b,
                                            uint32_t idesc,
                                            uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::discard [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_f8f6f4_ws(uint32_t tmem_base,
                                               uint64_t desc_a,
                                               uint64_t desc_b,
                                               uint32_t idesc,
                                               uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_i8_ws(uint32_t tmem_base,
                                           uint64_t desc_a,
                                           uint64_t desc_b,
                                           uint32_t idesc,
                                           uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::discard [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d)
      : "memory");
}

#define TCGEN05_ISSUE_WS_SP(KIND, BUFFER, OP, TARGET_TMEM, ENABLE_INPUT_D)                    \
  asm volatile(                                                                                \
      "{\n\t"                                                                                  \
      ".reg .pred p;\n\t"                                                                      \
      "setp.ne.b32 p, %5, 0;\n\t"                                                             \
      "tcgen05.mma.ws.sp.cta_group::1.kind::" KIND ".collector::" BUFFER                     \
      "::" OP " [%0], %1, %2, [%3], %4, p;\n\t"                                              \
      "}\n"                                                                                    \
      :                                                                                         \
      : "r"(TARGET_TMEM), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),        \
        "r"(ENABLE_INPUT_D)                                                                    \
      : "memory")

__device__ __forceinline__ void issue_tf32_ws_sparse(uint32_t tmem_base,
                                                     uint64_t desc_a,
                                                     uint64_t desc_b,
                                                     uint32_t sp_meta_tmem,
                                                     uint32_t idesc,
                                                     uint32_t enable_input_d,
                                                     bool fill,
                                                     uint32_t collector) {
  switch (collector & 3u) {
    case 0:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("tf32", "b0", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("tf32", "b0", "lastuse", tmem_base, enable_input_d);
      }
      break;
    case 1:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("tf32", "b1", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("tf32", "b1", "lastuse", tmem_base, enable_input_d);
      }
      break;
    case 2:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("tf32", "b2", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("tf32", "b2", "lastuse", tmem_base, enable_input_d);
      }
      break;
    default:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("tf32", "b3", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("tf32", "b3", "lastuse", tmem_base, enable_input_d);
      }
      break;
  }
}

__device__ __forceinline__ void issue_f16_ws_sparse(uint32_t tmem_base,
                                                    uint64_t desc_a,
                                                    uint64_t desc_b,
                                                    uint32_t sp_meta_tmem,
                                                    uint32_t idesc,
                                                    uint32_t enable_input_d,
                                                    bool fill,
                                                    uint32_t collector) {
  switch (collector & 3u) {
    case 0:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f16", "b0", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f16", "b0", "lastuse", tmem_base, enable_input_d);
      }
      break;
    case 1:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f16", "b1", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f16", "b1", "lastuse", tmem_base, enable_input_d);
      }
      break;
    case 2:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f16", "b2", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f16", "b2", "lastuse", tmem_base, enable_input_d);
      }
      break;
    default:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f16", "b3", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f16", "b3", "lastuse", tmem_base, enable_input_d);
      }
      break;
  }
}

__device__ __forceinline__ void issue_f8f6f4_ws_sparse(uint32_t tmem_base,
                                                       uint64_t desc_a,
                                                       uint64_t desc_b,
                                                       uint32_t sp_meta_tmem,
                                                       uint32_t idesc,
                                                       uint32_t enable_input_d,
                                                       bool fill,
                                                       uint32_t collector) {
  switch (collector & 3u) {
    case 0:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b0", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b0", "lastuse", tmem_base, enable_input_d);
      }
      break;
    case 1:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b1", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b1", "lastuse", tmem_base, enable_input_d);
      }
      break;
    case 2:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b2", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b2", "lastuse", tmem_base, enable_input_d);
      }
      break;
    default:
      if (fill) {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b3", "fill", tmem_base, enable_input_d);
      } else {
        TCGEN05_ISSUE_WS_SP("f8f6f4", "b3", "lastuse", tmem_base, enable_input_d);
      }
      break;
  }
}

#undef TCGEN05_ISSUE_WS_SP

#define TCGEN05_ISSUE_WS_SP(KIND, BUFFER, OP, TARGET_TMEM, ENABLE_INPUT_D)                    \
  asm volatile(                                                                                \
      "{\n\t"                                                                                  \
      ".reg .pred p;\n\t"                                                                      \
      "setp.ne.b32 p, %5, 0;\n\t"                                                             \
      "tcgen05.mma.ws.sp.cta_group::1.kind::" KIND ".collector::" BUFFER                     \
      "::" OP " [%0], %1, %2, [%3], %4, p;\n\t"                                              \
      "}\n"                                                                                    \
      :                                                                                         \
      : "r"(TARGET_TMEM), "l"(desc_a), "l"(desc_b), "r"(sp_meta_tmem), "r"(idesc),        \
        "r"(ENABLE_INPUT_D)                                                                    \
      : "memory")

__device__ __forceinline__ void issue_tf32_ws_sparse_discard(uint32_t tmem_base,
                                                             uint64_t desc_a,
                                                             uint64_t desc_b,
                                                             uint32_t sp_meta_tmem,
                                                             uint32_t idesc,
                                                             uint32_t enable_input_d,
                                                             uint32_t collector) {
#define TCGEN05_ISSUE_TF32_WS_SP_DISCARD(BUFFER)                                             \
  TCGEN05_ISSUE_WS_SP("tf32", BUFFER, "discard", tmem_base, enable_input_d)
  switch (collector & 3u) {
    case 0:
      TCGEN05_ISSUE_TF32_WS_SP_DISCARD("b0");
      break;
    case 1:
      TCGEN05_ISSUE_TF32_WS_SP_DISCARD("b1");
      break;
    case 2:
      TCGEN05_ISSUE_TF32_WS_SP_DISCARD("b2");
      break;
    default:
      TCGEN05_ISSUE_TF32_WS_SP_DISCARD("b3");
      break;
  }
#undef TCGEN05_ISSUE_TF32_WS_SP_DISCARD
}

__device__ __forceinline__ void issue_f8f6f4_ws_sparse_discard(uint32_t tmem_base,
                                                               uint64_t desc_a,
                                                               uint64_t desc_b,
                                                               uint32_t sp_meta_tmem,
                                                               uint32_t idesc,
                                                               uint32_t enable_input_d,
                                                               uint32_t collector) {
#define TCGEN05_ISSUE_F8F6F4_WS_SP_DISCARD(BUFFER)                                           \
  TCGEN05_ISSUE_WS_SP("f8f6f4", BUFFER, "discard", tmem_base, enable_input_d)
  switch (collector & 3u) {
    case 0:
      TCGEN05_ISSUE_F8F6F4_WS_SP_DISCARD("b0");
      break;
    case 1:
      TCGEN05_ISSUE_F8F6F4_WS_SP_DISCARD("b1");
      break;
    case 2:
      TCGEN05_ISSUE_F8F6F4_WS_SP_DISCARD("b2");
      break;
    default:
      TCGEN05_ISSUE_F8F6F4_WS_SP_DISCARD("b3");
      break;
  }
#undef TCGEN05_ISSUE_F8F6F4_WS_SP_DISCARD
}

__device__ __forceinline__ void issue_i8_ws_sparse_discard(uint32_t tmem_base,
                                                           uint64_t desc_a,
                                                           uint64_t desc_b,
                                                           uint32_t sp_meta_tmem,
                                                           uint32_t idesc,
                                                           uint32_t enable_input_d,
                                                           uint32_t collector) {
#define TCGEN05_ISSUE_I8_WS_SP_DISCARD(BUFFER)                                               \
  TCGEN05_ISSUE_WS_SP("i8", BUFFER, "discard", tmem_base, enable_input_d)
  switch (collector & 3u) {
    case 0:
      TCGEN05_ISSUE_I8_WS_SP_DISCARD("b0");
      break;
    case 1:
      TCGEN05_ISSUE_I8_WS_SP_DISCARD("b1");
      break;
    case 2:
      TCGEN05_ISSUE_I8_WS_SP_DISCARD("b2");
      break;
    default:
      TCGEN05_ISSUE_I8_WS_SP_DISCARD("b3");
      break;
  }
#undef TCGEN05_ISSUE_I8_WS_SP_DISCARD
}

#undef TCGEN05_ISSUE_WS_SP

__device__ __forceinline__ void fill_tf32(uint8_t* smem,
                                          int extent,
                                          const uint32_t* values,
                                          int k_count,
                                          const ProbeCase* pc,
                                          uint32_t operand_format,
                                          bool matrix_a,
                                          uint32_t row_base) {
  uint32_t* typed = reinterpret_cast<uint32_t*>(smem);
  for (int index = threadIdx.x; index < extent; index += kThreads) {
    const int logical_index = matrix_a ? index + static_cast<int>(row_base) : index;
    for (int k = 0; k < k_count; ++k) {
      typed[k_major_offset_for(pc, index, k, 4, k_count)] =
          operand_word(pc, values, operand_format, logical_index, k, matrix_a);
    }
  }
}

__device__ __forceinline__ void fill_f16(uint8_t* smem,
                                         int extent,
                                         const uint32_t* values,
                                         bool bf16,
                                         const ProbeCase* pc,
                                         uint32_t operand_format,
                                         bool matrix_a,
                                         uint32_t row_base) {
  uint16_t* typed = reinterpret_cast<uint16_t*>(smem);
  for (int index = threadIdx.x; index < extent; index += kThreads) {
    const int logical_index = matrix_a ? index + static_cast<int>(row_base) : index;
    for (int k = 0; k < 16; ++k) {
      const uint32_t value = operand_word(pc, values, operand_format, logical_index, k, matrix_a);
      typed[k_major_offset_for(pc, index, k, 8, 16)] =
          bf16 ? static_cast<uint16_t>(value >> 16) : static_cast<uint16_t>(value);
    }
  }
}

__device__ __forceinline__ void fill_f8f6f4_dense_operand(uint8_t* smem,
                                                          int extent,
                                                      const uint32_t* values,
                                                      uint32_t format,
                                                      const ProbeCase* pc,
                                                      bool matrix_a,
                                                      uint32_t row_base) {
  const uint32_t type_code = f8f6f4_type_code(format);
  const int storage_k = (pc->flags & kCoordFlags) != 0 ? f8f6f4_coordinate_storage_k(format)
                                                       : f8f6f4_storage_k(format);
  for (int index = threadIdx.x; index < extent; index += kThreads) {
    const int logical_index = matrix_a ? index + static_cast<int>(row_base) : index;
    if (type_code == 3 || type_code == 4) {
      for (int block = 0; block < 32; block += 16) {
        for (int group = 0; group < 4; ++group) {
          const int base = block + group * 4;
          const int byte_base = block + group * 3;
          const uint32_t v0 =
              operand_word(pc, values, format, logical_index, base + 0, matrix_a) & 0x3fu;
          const uint32_t v1 =
              operand_word(pc, values, format, logical_index, base + 1, matrix_a) & 0x3fu;
          const uint32_t v2 =
              operand_word(pc, values, format, logical_index, base + 2, matrix_a) & 0x3fu;
          const uint32_t v3 =
              operand_word(pc, values, format, logical_index, base + 3, matrix_a) & 0x3fu;
          smem[k_major_offset_for(pc, index, byte_base + 0, 16, storage_k)] =
              pack_f6_byte0(v0, v1);
          smem[k_major_offset_for(pc, index, byte_base + 1, 16, storage_k)] =
              pack_f6_byte1(v1, v2);
          smem[k_major_offset_for(pc, index, byte_base + 2, 16, storage_k)] =
              pack_f6_byte2(v2, v3);
        }
      }
    } else if (type_code == 5) {
      for (int block = 0; block < 32; block += 16) {
        for (int pair = 0; pair < 8; ++pair) {
          const int base = block + pair * 2;
          const int byte_k = block + pair;
          smem[k_major_offset_for(pc, index, byte_k, 16, storage_k)] =
              pack_f4_pair(operand_word(pc, values, format, logical_index, base + 0, matrix_a),
                           operand_word(pc, values, format, logical_index, base + 1, matrix_a));
        }
      }
    } else {
      for (int k = 0; k < 32; ++k) {
        smem[k_major_offset_for(pc, index, k, 16, storage_k)] =
            static_cast<uint8_t>(operand_word(pc, values, format, logical_index, k, matrix_a));
      }
    }
  }
}

__device__ __forceinline__ void fill_i8_dense_operand(uint8_t* smem,
                                                      int extent,
                                                      const uint32_t* values,
                                                      const ProbeCase* pc,
                                                      uint32_t operand_format,
                                                      bool matrix_a,
                                                      uint32_t row_base) {
  for (int index = threadIdx.x; index < extent; index += kThreads) {
    const int logical_index = matrix_a ? index + static_cast<int>(row_base) : index;
    for (int k = 0; k < 32; ++k) {
      smem[k_major_offset_for(pc, index, k, 16, 32)] =
          static_cast<uint8_t>(operand_word(pc, values, operand_format, logical_index, k, matrix_a));
    }
  }
}

__device__ __forceinline__ void clear_and_fill_operands(SharedStorage& shared,
                                                        const ProbeCase* pc,
                                                        uint32_t format,
                                                        uint32_t m,
                                                        uint32_t n,
                                                        int tid,
                                                        uint32_t row_base) {
  for (int i = tid; i < static_cast<int>(sizeof(shared.a_smem)); i += kThreads) {
    shared.a_smem[i] = 0;
  }
  for (int i = tid; i < static_cast<int>(sizeof(shared.b_smem)); i += kThreads) {
    shared.b_smem[i] = 0;
  }
  __syncthreads();

  if (format == kFmtTf32) {
    fill_tf32(shared.a_smem, m, pc->a, 8, pc, pc->a_format, true, row_base);
    fill_tf32(shared.b_smem, n, pc->b, 8, pc, pc->b_format, false, 0);
  } else if (format == kFmtBf16 || format == kFmtF16) {
    fill_f16(shared.a_smem, m, pc->a, pc->a_format == kFmtBf16, pc, pc->a_format, true, row_base);
    fill_f16(shared.b_smem, n, pc->b, pc->b_format == kFmtBf16, pc, pc->b_format, false, 0);
  } else if (format == kFmtI8) {
    fill_i8_dense_operand(shared.a_smem, m, pc->a, pc, pc->a_format, true, row_base);
    fill_i8_dense_operand(shared.b_smem, n, pc->b, pc, pc->b_format, false, 0);
  } else {
    fill_f8f6f4_dense_operand(shared.a_smem, m, pc->a, pc->a_format, pc, true, row_base);
    fill_f8f6f4_dense_operand(shared.b_smem, n, pc->b, pc->b_format, pc, false, 0);
  }
  __syncthreads();
  proxy_async_fence_shared_cta();
}

__device__ __forceinline__ int sparse_logical_k(uint32_t format,
                                                uint32_t metadata,
                                                uint32_t metadata_hi,
                                                int packed_k) {
  if (format == kFmtTf32) {
    const uint32_t selector = (metadata >> ((packed_k & 7) * 4)) & 0xFu;
    return packed_k * 2 + (selector == 0xEu ? 1 : 0);
  }
  const int chunk = packed_k >> 1;
  const uint32_t word = chunk < 8 ? metadata : metadata_hi;
  const uint32_t selector = (word >> ((chunk & 7) * 4)) & 0xFu;
  const int selector_index = (packed_k & 1) == 0
                                 ? static_cast<int>(selector & 0x3u)
                                 : static_cast<int>((selector >> 2) & 0x3u);
  return chunk * 4 + selector_index;
}

__device__ __forceinline__ void clear_a_operand(SharedStorage& shared, int tid) {
  for (int i = tid; i < static_cast<int>(sizeof(shared.a_smem)); i += kThreads) {
    shared.a_smem[i] = 0;
  }
  __syncthreads();
  proxy_async_fence_shared_cta();
}

__device__ __forceinline__ void clear_and_fill_sparse_operands(SharedStorage& shared,
                                                               const ProbeCase* pc,
                                                               uint32_t format,
                                                               uint32_t m,
                                                               uint32_t n,
                                                               int tid,
                                                               uint8_t* b_smem,
                                                               int b_clear_bytes,
                                                               uint32_t row_base) {
  for (int i = tid; i < static_cast<int>(sizeof(shared.a_smem)); i += kThreads) {
    shared.a_smem[i] = 0;
  }
  for (int i = tid; i < b_clear_bytes; i += kThreads) {
    b_smem[i] = 0;
  }
  __syncthreads();

  if (format == kFmtTf32) {
    for (int row = tid; row < static_cast<int>(m); row += kThreads) {
      const int logical_row = row + static_cast<int>(row_base);
      for (int k = 0; k < 8; ++k) {
        reinterpret_cast<uint32_t*>(shared.a_smem)[k_major_offset_for(pc, row, k, 4, 8)] =
            operand_word(pc,
                         pc->a,
                         pc->a_format,
                         logical_row,
                         sparse_logical_k(format, pc->metadata, pc->metadata_hi, k),
                         true);
      }
    }
    for (int col = tid; col < static_cast<int>(n); col += kThreads) {
      for (int k = 0; k < 16; ++k) {
        reinterpret_cast<uint32_t*>(b_smem)[k_major_offset_for(pc, col, k, 4, 16)] =
            operand_word(pc, pc->b, pc->b_format, col, k, false);
      }
    }
  } else if (format == kFmtBf16 || format == kFmtF16) {
    uint16_t* a_typed = reinterpret_cast<uint16_t*>(shared.a_smem);
    uint16_t* b_typed = reinterpret_cast<uint16_t*>(b_smem);
    const bool a_bf16 = pc->a_format == kFmtBf16;
    const bool b_bf16 = pc->b_format == kFmtBf16;
    for (int row = tid; row < static_cast<int>(m); row += kThreads) {
      const int logical_row = row + static_cast<int>(row_base);
      for (int k = 0; k < 16; ++k) {
        const uint32_t value =
            operand_word(pc,
                         pc->a,
                         pc->a_format,
                         logical_row,
                         sparse_logical_k(format, pc->metadata, pc->metadata_hi, k),
                         true);
        a_typed[k_major_offset_for(pc, row, k, 8, 16)] =
            a_bf16 ? static_cast<uint16_t>(value >> 16) : static_cast<uint16_t>(value);
      }
    }
    for (int col = tid; col < static_cast<int>(n); col += kThreads) {
      for (int k = 0; k < 32; ++k) {
        const uint32_t value = operand_word(pc, pc->b, pc->b_format, col, k, false);
        b_typed[k_major_offset_for(pc, col, k, 8, 32)] =
            b_bf16 ? static_cast<uint16_t>(value >> 16) : static_cast<uint16_t>(value);
      }
    }
  } else if (format == kFmtI8) {
    for (int row = tid; row < static_cast<int>(m); row += kThreads) {
      const int logical_row = row + static_cast<int>(row_base);
      for (int k = 0; k < 32; ++k) {
        shared.a_smem[k_major_offset_for(pc, row, k, 16, 32)] =
            static_cast<uint8_t>(
                operand_word(pc,
                             pc->a,
                             pc->a_format,
                             logical_row,
                             sparse_logical_k(format, pc->metadata, pc->metadata_hi, k),
                             true));
      }
    }
    for (int col = tid; col < static_cast<int>(n); col += kThreads) {
      for (int k = 0; k < 64; ++k) {
        b_smem[k_major_offset_for(pc, col, k, 16, 64)] =
            static_cast<uint8_t>(operand_word(pc, pc->b, pc->b_format, col, k, false));
      }
    }
  } else {
    const uint32_t a_type_code = f8f6f4_type_code(pc->a_format);
    const uint32_t b_type_code = f8f6f4_type_code(pc->b_format);
    const int a_storage_k = operand_coordinate_storage_k(format, pc->a_format, true, true);
    const int b_storage_k = operand_coordinate_storage_k(format, pc->b_format, true, false);
    for (int row = tid; row < static_cast<int>(m); row += kThreads) {
      const int logical_row = row + static_cast<int>(row_base);
      if (a_type_code == 3 || a_type_code == 4) {
        for (int block = 0; block < 32; block += 16) {
          for (int group = 0; group < 4; ++group) {
            const int base = block + group * 4;
            const int byte_base = block + group * 3;
            const uint32_t v0 =
                operand_word(pc,
                             pc->a,
                             pc->a_format,
                             logical_row,
                             sparse_logical_k(format, pc->metadata, pc->metadata_hi, base + 0),
                             true) &
                0x3fu;
            const uint32_t v1 =
                operand_word(pc,
                             pc->a,
                             pc->a_format,
                             logical_row,
                             sparse_logical_k(format, pc->metadata, pc->metadata_hi, base + 1),
                             true) &
                0x3fu;
            const uint32_t v2 =
                operand_word(pc,
                             pc->a,
                             pc->a_format,
                             logical_row,
                             sparse_logical_k(format, pc->metadata, pc->metadata_hi, base + 2),
                             true) &
                0x3fu;
            const uint32_t v3 =
                operand_word(pc,
                             pc->a,
                             pc->a_format,
                             logical_row,
                             sparse_logical_k(format, pc->metadata, pc->metadata_hi, base + 3),
                             true) &
                0x3fu;
            shared.a_smem[k_major_offset_for(pc, row, byte_base + 0, 16, a_storage_k)] =
                pack_f6_byte0(v0, v1);
            shared.a_smem[k_major_offset_for(pc, row, byte_base + 1, 16, a_storage_k)] =
                pack_f6_byte1(v1, v2);
            shared.a_smem[k_major_offset_for(pc, row, byte_base + 2, 16, a_storage_k)] =
                pack_f6_byte2(v2, v3);
          }
        }
      } else if (a_type_code == 5) {
        for (int block = 0; block < 32; block += 16) {
          for (int pair = 0; pair < 8; ++pair) {
            const int base = block + pair * 2;
            const int byte_k = block + pair;
            shared.a_smem[k_major_offset_for(pc, row, byte_k, 16, a_storage_k)] =
                pack_f4_pair(
                    operand_word(pc,
                                 pc->a,
                                 pc->a_format,
                                 logical_row,
                                 sparse_logical_k(format, pc->metadata, pc->metadata_hi, base + 0),
                                 true),
                    operand_word(pc,
                                 pc->a,
                                 pc->a_format,
                                 logical_row,
                                 sparse_logical_k(format, pc->metadata, pc->metadata_hi, base + 1),
                                 true));
          }
        }
      } else {
        for (int k = 0; k < 32; ++k) {
          shared.a_smem[k_major_offset_for(pc, row, k, 16, a_storage_k)] =
              static_cast<uint8_t>(
                  operand_word(pc,
                               pc->a,
                               pc->a_format,
                               logical_row,
                               sparse_logical_k(format, pc->metadata, pc->metadata_hi, k),
                               true));
        }
      }
    }
    for (int col = tid; col < static_cast<int>(n); col += kThreads) {
      if (b_type_code == 3 || b_type_code == 4) {
        for (int block = 0; block < 64; block += 16) {
          for (int group = 0; group < 4; ++group) {
            const int base = block + group * 4;
            const int byte_base = block + group * 3;
            const uint32_t v0 = operand_word(pc, pc->b, pc->b_format, col, base + 0, false) & 0x3fu;
            const uint32_t v1 = operand_word(pc, pc->b, pc->b_format, col, base + 1, false) & 0x3fu;
            const uint32_t v2 = operand_word(pc, pc->b, pc->b_format, col, base + 2, false) & 0x3fu;
            const uint32_t v3 = operand_word(pc, pc->b, pc->b_format, col, base + 3, false) & 0x3fu;
            b_smem[k_major_offset_for(pc, col, byte_base + 0, 16, b_storage_k)] =
                pack_f6_byte0(v0, v1);
            b_smem[k_major_offset_for(pc, col, byte_base + 1, 16, b_storage_k)] =
                pack_f6_byte1(v1, v2);
            b_smem[k_major_offset_for(pc, col, byte_base + 2, 16, b_storage_k)] =
                pack_f6_byte2(v2, v3);
          }
        }
      } else if (b_type_code == 5) {
        for (int block = 0; block < 64; block += 16) {
          for (int pair = 0; pair < 8; ++pair) {
            const int base = block + pair * 2;
            const int byte_k = block + pair;
            b_smem[k_major_offset_for(pc, col, byte_k, 16, b_storage_k)] =
                pack_f4_pair(operand_word(pc, pc->b, pc->b_format, col, base + 0, false),
                             operand_word(pc, pc->b, pc->b_format, col, base + 1, false));
          }
        }
      } else {
        for (int k = 0; k < 64; ++k) {
          b_smem[k_major_offset_for(pc, col, k, 16, b_storage_k)] =
              static_cast<uint8_t>(operand_word(pc, pc->b, pc->b_format, col, k, false));
        }
      }
    }
  }
  __syncthreads();
  proxy_async_fence_shared_cta();
}

template <uint32_t SparseMetadataOffsetCg1, bool CopyCase>
__global__ __launch_bounds__(kThreads) void tcgen05_shape_kernel(const ProbeCase* __restrict__ cases,
                                                                uint32_t* __restrict__ outputs,
                                                                size_t count) {
  __shared__ SharedStorage shared;

  const size_t first_case_idx = static_cast<size_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  if (first_case_idx >= count || tid >= static_cast<int>(kThreads)) {
    return;
  }

  if (warp == 0) {
    const uint32_t dst = smem_u32(&shared.tmem_base);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(dst), "r"(kTmemColumns)
        : "memory");
  }
  __syncthreads();

  const uint32_t tbase = shared.tmem_base;
  const uint32_t warp_taddr = tbase + static_cast<uint32_t>(warp * 32);
  const uint32_t ws_scratch_taddr = tbase + kWsScratchOffset;

  for (size_t case_idx = first_case_idx; case_idx < count;
       case_idx += static_cast<size_t>(gridDim.x)) {
    ProbeCase pc_storage{};
    const ProbeCase* pc = &cases[case_idx];
    if constexpr (CopyCase) {
      pc_storage = cases[case_idx];
      pc = &pc_storage;
    }
    const uint32_t format = pc->format;
    const uint32_t a_format = pc->a_format;
    const uint32_t b_format = pc->b_format;
    const uint32_t d_type = pc->d_type;
    const uint32_t m = pc->m;
    const uint32_t n = pc->n;
    const uint32_t c = (pc->flags & kCoordFlags) != 0 ? 0u : pc->c;
    const bool ws = (pc->flags & kFlagWs) != 0;
    const bool sparse = (pc->flags & kFlagSparse) != 0;
    const bool coordinate_mode = (pc->flags & kCoordFlags) != 0;
    const bool coordinate_random = (pc->flags & kFlagCoordRandom) != 0;
    const uint32_t sp_meta_base = tbase + SparseMetadataOffsetCg1;
    const bool ws_prefill = sparse && ws && (format == kFmtBf16 || format == kFmtF16);
    const int b_slot_offset =
        (sparse && ws) ? static_cast<int>((case_idx & 3ull) * kWsBSlotBytes) : 0;
    uint8_t* b_smem_case = shared.b_smem + b_slot_offset;

    if (tid == 0) {
      init_barrier(&shared.mma_barrier, 1);
      init_barrier(&shared.collector_barrier, 1);
    }
    __syncthreads();

    if (sparse) {
      clear_and_fill_sparse_operands(
          shared, pc, format, m, n, tid, b_smem_case, kWsBSlotBytes, 0);
    } else {
      clear_and_fill_operands(shared, pc, format, m, n, tid, 0);
    }

    if ((pc->flags & kCoordFlags) != 0) {
      const uint32_t store_groups = coordinate_load_groups(m, n, pc->flags);
      for (uint32_t group = 0; group < store_groups; ++group) {
        const uint32_t store_taddr = coordinate_tmem_addr(tbase, warp, group, m, n, pc->flags);
        if (coordinate_random) {
          const uint32_t physical_thread = static_cast<uint32_t>(tid);
          tmem_store_16x256_words(
              store_taddr,
              c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 0), d_type),
              c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 1), d_type),
              c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 2), d_type),
              c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 3), d_type));
        } else {
          tmem_store_16x256(store_taddr, c_store_word(c, d_type));
        }
      }
      tmem_wait_st();
    } else {
      tmem_store_16x256(warp_taddr, c_store_word(c, d_type));
      tmem_wait_st();
    }

    if (sparse) {
      tmem_store_sparse_metadata(sp_meta_base + static_cast<uint32_t>(warp * 32),
                                 format,
                                 pc->metadata,
                                 pc->metadata_hi);
      tmem_wait_st();
    }

    if (ws_prefill) {
      clear_a_operand(shared, tid);
      tcgen05_fence_before_sync();
      __syncthreads();
      tcgen05_fence_after_sync();
      if (tid == 0) {
        const uint32_t collector = static_cast<uint32_t>(case_idx);
        if (format == kFmtTf32) {
          const uint64_t desc_a =
              make_smem_desc(shared.a_smem,
                             operand_lbo_bytes(format, a_format, coordinate_mode),
                             operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
          const uint64_t desc_b =
              make_smem_desc(b_smem_case,
                             operand_lbo_bytes(format, b_format, coordinate_mode),
                             operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
          const uint32_t idesc =
              make_ws_instr_desc(format, a_format, b_format, d_type, m, n) | (1u << 2);
          issue_tf32_ws_sparse(
              ws_scratch_taddr, desc_a, desc_b, sp_meta_base, idesc, 0, true, collector);
        } else if (format == kFmtBf16 || format == kFmtF16) {
          const uint64_t desc_a =
              make_smem_desc(shared.a_smem,
                             operand_lbo_bytes(format, a_format, coordinate_mode),
                             operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
          const uint64_t desc_b =
              make_smem_desc(b_smem_case,
                             operand_lbo_bytes(format, b_format, coordinate_mode),
                             operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
          const uint32_t idesc =
              make_ws_instr_desc(format, a_format, b_format, d_type, m, n) | (1u << 2);
          issue_f16_ws_sparse(
              ws_scratch_taddr, desc_a, desc_b, sp_meta_base, idesc, 0, true, collector);
        } else {
          const uint64_t desc_a =
              make_smem_desc(shared.a_smem,
                             operand_lbo_bytes(format, a_format, coordinate_mode),
                             operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
          const uint64_t desc_b =
              make_smem_desc(b_smem_case,
                             operand_lbo_bytes(format, b_format, coordinate_mode),
                             operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
          const uint32_t idesc =
              make_f8f6f4_ws_instr_desc(a_format, b_format, d_type, m, n) | (1u << 2);
          issue_f8f6f4_ws_sparse(
              ws_scratch_taddr, desc_a, desc_b, sp_meta_base, idesc, 0, true, collector);
        }
        tcgen05_commit(&shared.collector_barrier);
      }
      wait_barrier(&shared.collector_barrier, 0);
      tcgen05_fence_after_sync();
      __syncthreads();
      clear_and_fill_sparse_operands(
          shared, pc, format, m, n, tid, b_smem_case, kWsBSlotBytes, 0);
    }

    tcgen05_fence_before_sync();
    __syncthreads();
    tcgen05_fence_after_sync();

    if (tid == 0) {
      uint64_t desc_a = 0;
      uint64_t desc_b = 0;
      const uint32_t collector = static_cast<uint32_t>(case_idx);
      if (format == kFmtTf32) {
        desc_a = make_smem_desc(shared.a_smem,
                                operand_lbo_bytes(format, a_format, coordinate_mode),
                                operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
        desc_b = make_smem_desc(b_smem_case,
                                operand_lbo_bytes(format, b_format, coordinate_mode),
                                operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
        const uint32_t idesc =
            (ws ? make_ws_instr_desc(format, a_format, b_format, d_type, m, n)
                : make_instr_desc(format, a_format, b_format, d_type, m, n)) |
            (sparse ? (1u << 2) : 0u);
        if (ws) {
          if (sparse) {
            issue_tf32_ws_sparse_discard(tbase, desc_a, desc_b, sp_meta_base, idesc, 1, collector);
          } else {
            issue_tf32_ws(tbase, desc_a, desc_b, idesc, 1);
          }
        } else if (sparse) {
          issue_tf32_sparse(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        } else {
          issue_tf32(tbase, desc_a, desc_b, idesc, 1);
        }
      } else if (format == kFmtBf16 || format == kFmtF16) {
        desc_a = make_smem_desc(shared.a_smem,
                                operand_lbo_bytes(format, a_format, coordinate_mode),
                                operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
        desc_b = make_smem_desc(b_smem_case,
                                operand_lbo_bytes(format, b_format, coordinate_mode),
                                operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
        const uint32_t idesc =
            (ws ? make_ws_instr_desc(format, a_format, b_format, d_type, m, n)
                : make_instr_desc(format, a_format, b_format, d_type, m, n)) |
            (sparse ? (1u << 2) : 0u);
        if (ws) {
          if (sparse) {
            issue_f16_ws_sparse(tbase, desc_a, desc_b, sp_meta_base, idesc, 1, false, collector);
          } else {
            issue_f16_ws(tbase, desc_a, desc_b, idesc, 1);
          }
        } else if (sparse) {
          issue_f16_sparse(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        } else {
          issue_f16(tbase, desc_a, desc_b, idesc, 1);
        }
      } else if (format == kFmtI8) {
        desc_a = make_smem_desc(shared.a_smem,
                                operand_lbo_bytes(format, a_format, coordinate_mode),
                                operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
        desc_b = make_smem_desc(b_smem_case,
                                operand_lbo_bytes(format, b_format, coordinate_mode),
                                operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
        const uint32_t idesc =
            (ws ? make_i8_ws_instr_desc(a_format, b_format, pc->flags, m, n)
                : make_i8_instr_desc(a_format, b_format, pc->flags, m, n)) |
            (sparse ? (1u << 2) : 0u);
        if (ws) {
          if (sparse) {
            issue_i8_ws_sparse_discard(tbase, desc_a, desc_b, sp_meta_base, idesc, 1, collector);
          } else {
            issue_i8_ws(tbase, desc_a, desc_b, idesc, 1);
          }
        } else if (sparse) {
          issue_i8_sparse(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        } else {
          issue_i8(tbase, desc_a, desc_b, idesc, 1);
        }
      } else {
        desc_a = make_smem_desc(shared.a_smem,
                                operand_lbo_bytes(format, a_format, coordinate_mode),
                                operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
        desc_b = make_smem_desc(b_smem_case,
                                operand_lbo_bytes(format, b_format, coordinate_mode),
                                operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
        const uint32_t idesc =
            (ws ? make_f8f6f4_ws_instr_desc(a_format, b_format, d_type, m, n)
                : make_f8f6f4_instr_desc_for_types(a_format, b_format, d_type, m, n)) |
            (sparse ? (1u << 2) : 0u);
        if (ws) {
          if (sparse) {
            issue_f8f6f4_ws_sparse_discard(
                tbase, desc_a, desc_b, sp_meta_base, idesc, 1, collector);
          } else {
            issue_f8f6f4_ws(tbase, desc_a, desc_b, idesc, 1);
          }
        } else if (sparse) {
          issue_f8f6f4_sparse(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        } else {
          issue_f8f6f4(tbase, desc_a, desc_b, idesc, 1);
        }
      }
      tcgen05_commit(&shared.mma_barrier);
    }

    wait_barrier(&shared.mma_barrier, 0);
    tcgen05_fence_after_sync();
    __syncthreads();

    const uint32_t load_groups = coordinate_mode ? coordinate_load_groups(m, n, pc->flags) : 1u;
    for (uint32_t group = 0; group < load_groups; ++group) {
      uint32_t r0 = 0;
      uint32_t r1 = 0;
      uint32_t r2 = 0;
      uint32_t r3 = 0;
      const uint32_t load_taddr =
          coordinate_mode ? coordinate_tmem_addr(tbase, warp, group, m, n, pc->flags) : warp_taddr;
      tmem_load_16x256(load_taddr, r0, r1, r2, r3);
      tmem_wait_ld();

      const size_t base =
          ((case_idx * static_cast<size_t>(load_groups) * kThreads) +
           static_cast<size_t>(group) * kThreads + static_cast<size_t>(tid)) *
          kRegsPerThread;
      outputs[base + 0] = r0;
      outputs[base + 1] = r1;
      outputs[base + 2] = r2;
      outputs[base + 3] = r3;
    }

    __syncthreads();
  }

  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
        :
        : "r"(tbase), "r"(kTmemColumns)
        : "memory");
  }
}

__global__ __launch_bounds__(kThreads) void tcgen05_shape_cg2_kernel(
    const ProbeCase* __restrict__ cases,
    uint32_t* __restrict__ outputs,
    size_t count) {
  __shared__ SharedStorage shared;

  const size_t case_idx = static_cast<size_t>(blockIdx.x >> 1);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  uint32_t cluster_rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cluster_rank));
  const uint32_t pair_rank = cluster_rank & 1u;
  if (case_idx >= count || tid >= static_cast<int>(kThreads)) {
    return;
  }

  const ProbeCase* pc = &cases[case_idx];
  const uint32_t format = pc->format;
  const uint32_t a_format = pc->a_format;
  const uint32_t b_format = pc->b_format;
  const uint32_t d_type = pc->d_type;
  const uint32_t m = pc->m;
  const uint32_t n = pc->n;
  const uint32_t c = (pc->flags & kCoordFlags) != 0 ? 0u : pc->c;
  const bool sparse = (pc->flags & kFlagSparse) != 0;
  const bool coordinate_mode = (pc->flags & kCoordFlags) != 0;
  const bool coordinate_random = (pc->flags & kFlagCoordRandom) != 0;

  if (tid == 0) {
    init_barrier(&shared.mma_barrier, 1);
  }
  __syncthreads();

  if (warp == 0) {
    const uint32_t dst = smem_u32(&shared.tmem_base);
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(dst), "r"(kTmemColumns)
        : "memory");
  }
  __syncthreads();
  cluster_sync();

  const uint32_t row_base = coordinate_cta_row_base(m, pc->flags, pair_rank);
  const uint32_t local_m = coordinate_mode ? (m / 2u) : m;
  const uint32_t idesc_m = m;
  const uint32_t idesc_n = n < 256u ? n * 2u : n;
  if (sparse) {
    clear_and_fill_sparse_operands(
        shared, pc, format, local_m, n, tid, shared.b_smem, kWsBSlotBytes, row_base);
  } else {
    clear_and_fill_operands(shared, pc, format, local_m, n, tid, row_base);
  }

  const uint32_t tbase = shared.tmem_base;
  const uint32_t warp_taddr = tbase + static_cast<uint32_t>(warp * 32);

  if (coordinate_mode) {
    const uint32_t store_groups = coordinate_load_groups(m, n, pc->flags);
    for (uint32_t group = 0; group < store_groups; ++group) {
      const uint32_t store_taddr = coordinate_tmem_addr(tbase, warp, group, m, n, pc->flags);
      if (coordinate_random) {
        const uint32_t physical_thread = pair_rank * kThreads + static_cast<uint32_t>(tid);
        tmem_store_16x256_words(
            store_taddr,
            c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 0), d_type),
            c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 1), d_type),
            c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 2), d_type),
            c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 3), d_type));
      } else {
        tmem_store_16x256(store_taddr, c_store_word(c, d_type));
      }
    }
    tmem_wait_st();
  } else {
    tmem_store_16x256(warp_taddr, c_store_word(c, d_type));
    tmem_wait_st();
  }

  const uint32_t sp_meta_base = tbase + kSparseMetadataOffsetCg2;
  if (sparse) {
    tmem_store_sparse_metadata(sp_meta_base + static_cast<uint32_t>(warp * 32),
                               format,
                               pc->metadata,
                               pc->metadata_hi);
    tmem_wait_st();
  }

  tcgen05_fence_before_sync();
  __syncthreads();
  cluster_sync();
  tcgen05_fence_after_sync();

  // cta_group::2 coordinate probes need a doubled N descriptor; N256 is split
  // into two legal N128 halves because the descriptor cannot encode 512.
  const bool split_n256 = coordinate_mode && n == 256u;
  const uint32_t d_high_tbase = tbase + 128u;

  if (pair_rank == 0 && tid == 0) {
    uint64_t desc_a = 0;
    uint64_t desc_b = 0;
    const uint32_t b_high_offset =
        split_n256 ? coordinate_b_column_byte_offset(pc, format, b_format, sparse, 128u) : 0u;
    if (format == kFmtTf32) {
      desc_a = make_smem_desc(shared.a_smem,
                              operand_lbo_bytes(format, a_format, coordinate_mode),
                              operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
      desc_b = make_smem_desc(shared.b_smem,
                              operand_lbo_bytes(format, b_format, coordinate_mode),
                              operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
      const uint64_t desc_b_high =
          split_n256 ? make_smem_desc(shared.b_smem + b_high_offset,
                                      operand_lbo_bytes(format, b_format, coordinate_mode),
                                      operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false))
                     : 0ull;
      const uint32_t idesc = make_instr_desc(format, a_format, b_format, d_type, idesc_m, idesc_n) |
                             (sparse ? (1u << 2) : 0u);
      if (sparse) {
        issue_tf32_sparse_cg2(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_tf32_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 0);
        }
      } else {
        issue_tf32_cg2(tbase, desc_a, desc_b, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_tf32_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 0);
        }
      }
    } else if (format == kFmtBf16 || format == kFmtF16) {
      desc_a = make_smem_desc(shared.a_smem,
                              operand_lbo_bytes(format, a_format, coordinate_mode),
                              operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
      desc_b = make_smem_desc(shared.b_smem,
                              operand_lbo_bytes(format, b_format, coordinate_mode),
                              operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
      const uint64_t desc_b_high =
          split_n256 ? make_smem_desc(shared.b_smem + b_high_offset,
                                      operand_lbo_bytes(format, b_format, coordinate_mode),
                                      operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false))
                     : 0ull;
      const uint32_t idesc = make_instr_desc(format, a_format, b_format, d_type, idesc_m, idesc_n) |
                             (sparse ? (1u << 2) : 0u);
      if (sparse) {
        issue_f16_sparse_cg2(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_f16_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 0);
        }
      } else {
        issue_f16_cg2(tbase, desc_a, desc_b, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_f16_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 0);
        }
      }
    } else if (format == kFmtI8) {
      desc_a = make_smem_desc(shared.a_smem,
                              operand_lbo_bytes(format, a_format, coordinate_mode),
                              operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
      desc_b = make_smem_desc(shared.b_smem,
                              operand_lbo_bytes(format, b_format, coordinate_mode),
                              operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
      const uint64_t desc_b_high =
          split_n256 ? make_smem_desc(shared.b_smem + b_high_offset,
                                      operand_lbo_bytes(format, b_format, coordinate_mode),
                                      operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false))
                     : 0ull;
      const uint32_t idesc = make_i8_instr_desc(a_format, b_format, pc->flags, idesc_m, idesc_n) |
                             (sparse ? (1u << 2) : 0u);
      if (sparse) {
        issue_i8_sparse_cg2(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_i8_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 0);
        }
      } else {
        issue_i8_cg2(tbase, desc_a, desc_b, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_i8_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 0);
        }
      }
    } else {
      desc_a = make_smem_desc(shared.a_smem,
                              operand_lbo_bytes(format, a_format, coordinate_mode),
                              operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
      desc_b = make_smem_desc(shared.b_smem,
                              operand_lbo_bytes(format, b_format, coordinate_mode),
                              operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));
      const uint64_t desc_b_high =
          split_n256 ? make_smem_desc(shared.b_smem + b_high_offset,
                                      operand_lbo_bytes(format, b_format, coordinate_mode),
                                      operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false))
                     : 0ull;
      const uint32_t idesc = make_f8f6f4_instr_desc_for_types(a_format, b_format, d_type, idesc_m, idesc_n) |
                             (sparse ? (1u << 2) : 0u);
      if (sparse) {
        issue_f8f6f4_sparse_cg2(tbase, desc_a, desc_b, sp_meta_base, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_f8f6f4_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 0);
        }
      } else {
        issue_f8f6f4_cg2(tbase, desc_a, desc_b, idesc, 1);
        if (split_n256 && !coordinate_random) {
          issue_f8f6f4_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 0);
        }
      }
    }
    tcgen05_commit_cg2(&shared.mma_barrier);
    wait_barrier(&shared.mma_barrier, 0);
  }

  if (split_n256 && coordinate_random) {
    tcgen05_fence_before_sync();
    __syncthreads();
    cluster_sync();
    tcgen05_fence_after_sync();

    const uint32_t store_groups = coordinate_load_groups(m, n, pc->flags);
    const uint32_t row_groups = coordinate_row_groups(m, pc->flags);
    for (uint32_t group = 0; group < store_groups; ++group) {
      const uint32_t col_group = group / row_groups;
      if (col_group < 16u) {
        continue;
      }
      const uint32_t store_taddr = coordinate_tmem_addr(tbase, warp, group, m, n, pc->flags);
      const uint32_t physical_thread = pair_rank * kThreads + static_cast<uint32_t>(tid);
      tmem_store_16x256_words(
          store_taddr,
          c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 0), d_type),
          c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 1), d_type),
          c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 2), d_type),
          c_store_word(coordinate_random_c_word(pc, d_type, group, physical_thread, 3), d_type));
    }
    tmem_wait_st();

    tcgen05_fence_before_sync();
    __syncthreads();
    cluster_sync();
    tcgen05_fence_after_sync();

    if (pair_rank == 0 && tid == 0) {
      const uint32_t b_high_offset =
          coordinate_b_column_byte_offset(pc, format, b_format, sparse, 128u);
      uint64_t desc_a = make_smem_desc(
          shared.a_smem,
          operand_lbo_bytes(format, a_format, coordinate_mode),
          operand_sbo_bytes(format, a_format, coordinate_mode, sparse, true));
      uint64_t desc_b_high = make_smem_desc(
          shared.b_smem + b_high_offset,
          operand_lbo_bytes(format, b_format, coordinate_mode),
          operand_sbo_bytes(format, b_format, coordinate_mode, sparse, false));

      if (format == kFmtTf32) {
        const uint32_t idesc =
            make_instr_desc(format, a_format, b_format, d_type, idesc_m, idesc_n) |
            (sparse ? (1u << 2) : 0u);
        if (sparse) {
          issue_tf32_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 1);
        } else {
          issue_tf32_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 1);
        }
      } else if (format == kFmtBf16 || format == kFmtF16) {
        const uint32_t idesc =
            make_instr_desc(format, a_format, b_format, d_type, idesc_m, idesc_n) |
            (sparse ? (1u << 2) : 0u);
        if (sparse) {
          issue_f16_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 1);
        } else {
          issue_f16_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 1);
        }
      } else if (format == kFmtI8) {
        const uint32_t idesc =
            make_i8_instr_desc(a_format, b_format, pc->flags, idesc_m, idesc_n) |
            (sparse ? (1u << 2) : 0u);
        if (sparse) {
          issue_i8_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 1);
        } else {
          issue_i8_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 1);
        }
      } else {
        const uint32_t idesc =
            make_f8f6f4_instr_desc_for_types(a_format, b_format, d_type, idesc_m, idesc_n) |
            (sparse ? (1u << 2) : 0u);
        if (sparse) {
          issue_f8f6f4_sparse_cg2(d_high_tbase, desc_a, desc_b_high, sp_meta_base, idesc, 1);
        } else {
          issue_f8f6f4_cg2(d_high_tbase, desc_a, desc_b_high, idesc, 1);
        }
      }
      tcgen05_commit_cg2(&shared.mma_barrier);
      wait_barrier(&shared.mma_barrier, 1);
    }
  }

  tcgen05_fence_before_sync();
  cluster_sync();
  tcgen05_fence_after_sync();
  __syncthreads();

  const uint32_t load_groups =
      coordinate_mode ? coordinate_load_groups(m, n, pc->flags) : 1u;
  for (uint32_t group = 0; group < load_groups; ++group) {
    uint32_t r0 = 0;
    uint32_t r1 = 0;
    uint32_t r2 = 0;
    uint32_t r3 = 0;
    const uint32_t load_taddr =
        coordinate_mode ? coordinate_tmem_addr(tbase, warp, group, m, n, pc->flags) : warp_taddr;
    tmem_load_16x256(load_taddr, r0, r1, r2, r3);
    tmem_wait_ld();
    if (coordinate_mode && !sparse && m == 128u && tid >= 64) {
      r0 = 0;
      r1 = 0;
      r2 = 0;
      r3 = 0;
    }

    const size_t base =
        ((case_idx * static_cast<size_t>(load_groups) * kCtaGroup2Threads) +
         static_cast<size_t>(group) * kCtaGroup2Threads +
         static_cast<size_t>(pair_rank) * kThreads + static_cast<size_t>(tid)) *
        kRegsPerThread;
    outputs[base + 0] = r0;
    outputs[base + 1] = r1;
    outputs[base + 2] = r2;
    outputs[base + 3] = r3;
  }

  __syncthreads();
  cluster_sync();
  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::: "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
        :
        : "r"(tbase), "r"(kTmemColumns)
        : "memory");
  }
}

std::vector<uint32_t> run_cases(const std::vector<ProbeCase>& cases, int device) {
  die_cuda(cudaSetDevice(device), "cudaSetDevice");

  ProbeCase* d_cases = nullptr;
  uint32_t* d_outputs = nullptr;
  const bool cta_group2 = !cases.empty() && ((cases.front().flags & kFlagCtaGroup2) != 0);
  const bool sparse = !cases.empty() && ((cases.front().flags & kFlagSparse) != 0);
  const bool ws = !cases.empty() && ((cases.front().flags & kFlagWs) != 0);
  const bool coordinate_mode = !cases.empty() && ((cases.front().flags & kCoordFlags) != 0);
  const uint32_t format = cases.empty() ? 0u : cases.front().format;
  for (const ProbeCase& pc : cases) {
    if (pc.format != format) {
      throw std::runtime_error("cannot mix formats in one launch");
    }
    if (((pc.flags & kFlagCtaGroup2) != 0) != cta_group2) {
      throw std::runtime_error("cannot mix cta_group::1 and cta_group::2 cases in one launch");
    }
    if (((pc.flags & kCoordFlags) != 0) != coordinate_mode) {
      throw std::runtime_error("cannot mix coordinate and non-coordinate cases in one launch");
    }
    if (((pc.flags & kFlagSparse) != 0) != sparse) {
      throw std::runtime_error("cannot mix dense and sparse cases in one launch");
    }
    if (((pc.flags & kFlagWs) != 0) != ws) {
      throw std::runtime_error("cannot mix .ws and non-.ws cases in one launch");
    }
  }
  const uint32_t threads_per_case = output_threads_per_case(cases);
  const ProbeCase* host_cases = cases.data();
  const size_t launch_count = cases.size();
  const size_t case_bytes = sizeof(ProbeCase) * launch_count;
  const size_t returned_output_count = cases.size() * threads_per_case * kRegsPerThread;
  const size_t launch_output_count = launch_count * threads_per_case * kRegsPerThread;
  const size_t output_bytes = sizeof(uint32_t) * launch_output_count;

  if (!cases.empty()) {
    die_cuda(cudaMalloc(&d_cases, case_bytes), "cudaMalloc cases");
    die_cuda(cudaMalloc(&d_outputs, output_bytes), "cudaMalloc outputs");
    die_cuda(cudaMemcpy(d_cases, host_cases, case_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy cases");

    if (launch_count > static_cast<size_t>(0x3fffffff)) {
      throw std::runtime_error("too many cases for one launch");
    }
    if (cta_group2) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = 2;
      attr.val.clusterDim.y = 1;
      attr.val.clusterDim.z = 1;

      cudaLaunchConfig_t config{};
      config.gridDim = dim3(static_cast<unsigned int>(launch_count * 2), 1, 1);
      config.blockDim = dim3(kThreads, 1, 1);
      config.dynamicSmemBytes = 0;
      config.stream = nullptr;
      config.attrs = &attr;
      config.numAttrs = 1;
      die_cuda(cudaLaunchKernelEx(&config, tcgen05_shape_cg2_kernel, d_cases, d_outputs,
                                  launch_count),
               "kernel launch");
    } else {
      const unsigned int grid_blocks = sparse ? 1u : static_cast<unsigned int>(cases.size());
      if (sparse && (format == kFmtE4m3 || format == kFmtE5m2 ||
                     (coordinate_mode && format == kFmtE2m1))) {
        tcgen05_shape_kernel<kSparseMetadataOffsetF8F4Cg1, true><<<grid_blocks, kThreads>>>(
            d_cases, d_outputs, launch_count);
      } else if (sparse && format == kFmtE2m1) {
        tcgen05_shape_kernel<kSparseMetadataOffsetLowCg1, true><<<grid_blocks, kThreads>>>(
            d_cases, d_outputs, launch_count);
      } else {
        tcgen05_shape_kernel<kSparseMetadataOffsetCg1, false><<<grid_blocks, kThreads>>>(
            d_cases, d_outputs, launch_count);
      }
      die_cuda(cudaGetLastError(), "kernel launch");
    }
    die_cuda(cudaDeviceSynchronize(), "kernel sync");
  }

  std::vector<uint32_t> launch_outputs(launch_output_count);
  if (!launch_outputs.empty()) {
    die_cuda(cudaMemcpy(launch_outputs.data(), d_outputs, output_bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy outputs");
  }
  std::vector<uint32_t> outputs(returned_output_count);
  if (!outputs.empty()) {
    std::memcpy(outputs.data(), launch_outputs.data(), sizeof(uint32_t) * returned_output_count);
  }

  if (d_outputs) {
    cudaFree(d_outputs);
  }
  if (d_cases) {
    cudaFree(d_cases);
  }
  return outputs;
}

void print_info(int device) {
  die_cuda(cudaSetDevice(device), "cudaSetDevice");
  cudaDeviceProp prop{};
  die_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  std::cout << "{"
            << "\"device\":" << device << ","
            << "\"name\":\"" << prop.name << "\","
            << "\"compute_capability\":\"" << prop.major << "." << prop.minor << "\","
            << "\"op\":\"tcgen05.mma(.sp,.ws).cta_group::{1,2} runtime-shape\","
            << "\"formats\":\"tf32,bf16,f16,e4m3,e5m2,e2m3,e3m2,e2m1,i8\","
            << "\"m_values\":\"cg1 non-ws:64,128; cg1 ws:32,64,128; cg2:128,256\","
            << "\"n_range\":\"cg1 non-ws:8..256 step 8; cg1 ws:64,128,256; cg2 dense/sparse:16..256 step 16\","
            << "\"threads\":" << kThreads << ","
            << "\"regs_per_thread\":" << kRegsPerThread << "}" << std::endl;
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --input cases.bin --output outputs.bin [--device N]\n"
            << "       " << argv0 << " --info [--device N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string input_path;
    std::string output_path;
    int device = 0;
    bool info = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--input" && i + 1 < argc) {
        input_path = argv[++i];
      } else if (arg == "--output" && i + 1 < argc) {
        output_path = argv[++i];
      } else if (arg == "--device" && i + 1 < argc) {
        device = std::atoi(argv[++i]);
      } else if (arg == "--info") {
        info = true;
      } else if (arg == "--help" || arg == "-h") {
        usage(argv[0]);
        return 0;
      } else {
        usage(argv[0]);
        return 2;
      }
    }

    if (info) {
      print_info(device);
      return 0;
    }

    if (input_path.empty() || output_path.empty()) {
      usage(argv[0]);
      return 2;
    }

    auto cases = read_cases(input_path);
    const uint32_t threads_per_case = output_threads_per_case(cases);
    auto outputs = run_cases(cases, device);
    write_outputs(output_path, outputs, cases.size(), threads_per_case);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "tcgen05_shape_probe: " << ex.what() << "\n";
    return 1;
  }
}
