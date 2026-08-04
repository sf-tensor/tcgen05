// Binary harness for block-scaled tcgen05 paths not covered by the original
// NVFP4 UE4M3 runner:
//   tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
//   tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
//   tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X
//   tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
//
// The kernel maps one scalar dot-add onto output D[0,0] by filling row 0 of A
// and column 0 of B. Inputs are uint32 containers; the runner stores the low
// format bits into the PTX shared-memory packing expected by each MMA kind.

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
constexpr uint32_t kOpTcgen05Mxf8f6f4E4M3Ue8m0M128N8K32 = 20;
constexpr uint32_t kOpTcgen05Mxf8f6f4E5M2Ue8m0M128N8K32 = 21;
constexpr uint32_t kOpTcgen05Mxf8f6f4E2M3Ue8m0M128N8K32 = 22;
constexpr uint32_t kOpTcgen05Mxf8f6f4E3M2Ue8m0M128N8K32 = 23;
constexpr uint32_t kOpTcgen05Mxf8f6f4E2M1Ue8m0M128N8K32 = 24;
constexpr uint32_t kOpTcgen05Mxf4E2M1Ue8m0M128N8K64 = 25;
constexpr uint32_t kOpTcgen05Mxf4Nvfp4E2M1Ue8m0M128N8K64Scale2X = 26;
constexpr uint32_t kOpTcgen05Mxf4Nvfp4E2M1Ue8m0M128N8K64Scale4X = 27;
constexpr uint32_t kOpTcgen05Mxf4Nvfp4E2M1Ue4m3M128N8K64Scale4X = 37;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kRegsPerThread = 4;
constexpr uint32_t kTmemColumns = 512;
constexpr int kMaxK = 64;
constexpr int kMaxScaleCount = 4;
constexpr uint32_t kScaleAOffset = 128;
constexpr uint32_t kScaleBOffset = 256;

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
  uint32_t a[kMaxK];
  uint32_t b[kMaxK];
  uint32_t c;
  uint32_t scale_a[kMaxScaleCount];
  uint32_t scale_b[kMaxScaleCount];
  uint32_t a_format;
  uint32_t b_format;
  uint32_t n;
  uint32_t cta_group;
  uint32_t m;
};

struct InputBatch {
  uint32_t op = 0;
  std::vector<ProbeCase> cases;
};

struct alignas(128) SharedStorage {
  alignas(16) uint32_t tmem_base;
  alignas(8) uint64_t mma_barrier;
  alignas(128) uint8_t a_smem[4096];
  alignas(128) uint8_t b_smem[4096];
};

static_assert(sizeof(InputHeader) == 24, "unexpected InputHeader packing");
static_assert(sizeof(OutputHeader) == 32, "unexpected OutputHeader packing");
static_assert(sizeof(ProbeCase) == 568, "unexpected ProbeCase packing");

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

__host__ __device__ __forceinline__ bool is_mxf8f6f4_op(uint32_t op) {
  return op == kOpTcgen05Mxf8f6f4E4M3Ue8m0M128N8K32 ||
         op == kOpTcgen05Mxf8f6f4E5M2Ue8m0M128N8K32 ||
         op == kOpTcgen05Mxf8f6f4E2M3Ue8m0M128N8K32 ||
         op == kOpTcgen05Mxf8f6f4E3M2Ue8m0M128N8K32 ||
         op == kOpTcgen05Mxf8f6f4E2M1Ue8m0M128N8K32;
}

bool is_supported_op(uint32_t op) {
  return is_mxf8f6f4_op(op) || op == kOpTcgen05Mxf4E2M1Ue8m0M128N8K64 ||
         op == kOpTcgen05Mxf4Nvfp4E2M1Ue8m0M128N8K64Scale2X ||
         op == kOpTcgen05Mxf4Nvfp4E2M1Ue8m0M128N8K64Scale4X ||
         op == kOpTcgen05Mxf4Nvfp4E2M1Ue4m3M128N8K64Scale4X;
}

__host__ __device__ __forceinline__ uint32_t type_code_for_op(uint32_t op) {
  switch (op) {
    case kOpTcgen05Mxf8f6f4E4M3Ue8m0M128N8K32:
      return 0;  // E4M3
    case kOpTcgen05Mxf8f6f4E5M2Ue8m0M128N8K32:
      return 1;  // E5M2
    case kOpTcgen05Mxf8f6f4E2M3Ue8m0M128N8K32:
      return 3;  // E2M3
    case kOpTcgen05Mxf8f6f4E3M2Ue8m0M128N8K32:
      return 4;  // E3M2
    case kOpTcgen05Mxf8f6f4E2M1Ue8m0M128N8K32:
      return 5;  // E2M1 for mxf8f6f4 descriptor
    case kOpTcgen05Mxf4E2M1Ue8m0M128N8K64:
    case kOpTcgen05Mxf4Nvfp4E2M1Ue8m0M128N8K64Scale2X:
    case kOpTcgen05Mxf4Nvfp4E2M1Ue8m0M128N8K64Scale4X:
    case kOpTcgen05Mxf4Nvfp4E2M1Ue4m3M128N8K64Scale4X:
      return 1;  // E2M1 for mxf4/mxf4nvf4 descriptors
    default:
      return 0;
  }
}

__host__ __device__ __forceinline__ int k_for_op(uint32_t op) {
  return is_mxf8f6f4_op(op) ? 32 : 64;
}

InputBatch read_cases(const std::string& path) {
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
  if (!is_supported_op(header.op)) {
    throw std::runtime_error("unsupported op id");
  }
  if (header.count > (1ull << 34)) {
    throw std::runtime_error("case count is unreasonable");
  }

  InputBatch batch{};
  batch.op = header.op;
  batch.cases.resize(static_cast<size_t>(header.count));
  if (!batch.cases.empty()) {
    read_exact(in, batch.cases.data(), batch.cases.size(), "cases");
  }
  for (const ProbeCase& pc : batch.cases) {
    if (pc.m != 128 && pc.m != 256) {
      throw std::runtime_error("unsupported block-scaled M shape");
    }
    if (pc.cta_group == 1 && pc.m != 128) {
      throw std::runtime_error("unsupported cta_group::1 block-scaled M shape");
    }
    if (pc.n < 8 || pc.n > 256 || (pc.n % 8) != 0) {
      throw std::runtime_error("unsupported block-scaled N shape");
    }
    if (pc.cta_group != 1 && pc.cta_group != 2) {
      throw std::runtime_error("unsupported block-scaled cta_group");
    }
    if (pc.cta_group == 2 && (pc.n < 16 || (pc.n % 16) != 0)) {
      throw std::runtime_error("unsupported cta_group::2 block-scaled N shape");
    }
  }
  return batch;
}

void write_outputs(const std::string& path,
                   const std::vector<uint32_t>& outputs,
                   size_t count,
                   uint32_t op,
                   uint32_t lanes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }

  OutputHeader header{};
  std::memcpy(header.magic, kOutputMagic, sizeof(header.magic));
  header.version = kVersion;
  header.op = op;
  header.lanes = lanes;
  header.acc_regs = kRegsPerThread;
  header.count = count;

  write_exact(out, &header, 1, "output header");
  if (!outputs.empty()) {
    write_exact(out, outputs.data(), outputs.size(), "outputs");
  }
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
  desc |= 1ull << 46;  // descriptor version for Blackwell
  return desc;
}

__device__ __forceinline__ uint32_t make_mxf8f6f4_instr_desc(uint32_t a_type_code,
                                                             uint32_t b_type_code,
                                                             uint32_t n,
                                                             uint32_t m) {
  uint32_t desc = 0;
  desc |= (a_type_code & 0x7u) << 7;
  desc |= (b_type_code & 0x7u) << 10;
  desc |= (n >> 3) << 17;
  desc |= 1u << 23;  // scale matrix type: UE8M0
  desc |= (m >> 7) << 27;
  return desc;
}

__host__ __device__ __forceinline__ bool is_ue4m3_nvfp4_op(uint32_t op) {
  return op == kOpTcgen05Mxf4Nvfp4E2M1Ue4m3M128N8K64Scale4X;
}

__device__ __forceinline__ uint32_t make_mxf4_instr_desc(uint32_t n, uint32_t m, bool ue8m0_scales) {
  uint32_t desc = 0;
  desc |= 1u << 7;   // A type: E2M1
  desc |= 1u << 10;  // B type: E2M1
  desc |= (n >> 3) << 17;
  if (ue8m0_scales) {
    desc |= 1u << 23;  // Scale matrix type: UE8M0; clear selects UE4M3.
  }
  desc |= (m >> 7) << 27;
  return desc;
}

__device__ __forceinline__ uint8_t mxf8f6f4_from_container(uint32_t bits) {
  return static_cast<uint8_t>(bits & 0xFFu);
}

__device__ __forceinline__ uint8_t fp4_from_container(uint32_t bits) {
  return static_cast<uint8_t>(bits & 0xFu);
}

__device__ __forceinline__ uint8_t pack_f6_byte0(uint32_t v0, uint32_t v1) {
  return static_cast<uint8_t>((v0 & 0x3Fu) | ((v1 & 0x3u) << 6));
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

__device__ __forceinline__ uint32_t pack_scale_word(const uint32_t* scales) {
  return (scales[0] & 0xFFu) | ((scales[1] & 0xFFu) << 8) | ((scales[2] & 0xFFu) << 16) |
         ((scales[3] & 0xFFu) << 24);
}

__device__ __forceinline__ int k_major_mxf8f6f4_a_offset(int row, int k) {
  // K-major no-swizzle canonical layout for 8-bit tcgen05 operands.
  return (row & 7) * 16 + (row >> 3) * 32 + (k & 15) + (k >> 4) * 64;
}

__device__ __forceinline__ int k_major_mxf8f6f4_b_offset(int col, int k) {
  return (col & 7) * 16 + (col >> 3) * 32 + (k & 15) + (k >> 4) * 64;
}

__device__ __forceinline__ void fill_mxf8f6f4_a_operand(uint8_t* smem,
                                                        const uint32_t* values,
                                                        uint32_t type_code,
                                                        int rows) {
  for (int row = 0; row < rows; ++row) {
    if (type_code == 3 || type_code == 4) {
      for (int block = 0; block < 32; block += 16) {
        for (int group = 0; group < 4; ++group) {
          const int base = block + group * 4;
          const int byte_base = block + group * 3;
          const uint32_t v0 = values[base + 0] & 0x3Fu;
          const uint32_t v1 = values[base + 1] & 0x3Fu;
          const uint32_t v2 = values[base + 2] & 0x3Fu;
          const uint32_t v3 = values[base + 3] & 0x3Fu;
          smem[k_major_mxf8f6f4_a_offset(row, byte_base + 0)] = pack_f6_byte0(v0, v1);
          smem[k_major_mxf8f6f4_a_offset(row, byte_base + 1)] = pack_f6_byte1(v1, v2);
          smem[k_major_mxf8f6f4_a_offset(row, byte_base + 2)] = pack_f6_byte2(v2, v3);
        }
      }
    } else if (type_code == 5) {
      for (int block = 0; block < 32; block += 16) {
        for (int pair = 0; pair < 8; ++pair) {
          const int base = block + pair * 2;
          const int byte_k = block + pair;
          smem[k_major_mxf8f6f4_a_offset(row, byte_k)] = pack_f4_pair(values[base + 0], values[base + 1]);
        }
      }
    } else {
      for (int k = 0; k < 32; ++k) {
        smem[k_major_mxf8f6f4_a_offset(row, k)] = mxf8f6f4_from_container(values[k]);
      }
    }
  }
}

__device__ __forceinline__ void fill_mxf8f6f4_b_operand(uint8_t* smem,
                                                        const uint32_t* values,
                                                        uint32_t type_code,
                                                        int cols) {
  for (int col = 0; col < cols; ++col) {
    if (type_code == 3 || type_code == 4) {
      for (int block = 0; block < 32; block += 16) {
        for (int group = 0; group < 4; ++group) {
          const int base = block + group * 4;
          const int byte_base = block + group * 3;
          const uint32_t v0 = values[base + 0] & 0x3Fu;
          const uint32_t v1 = values[base + 1] & 0x3Fu;
          const uint32_t v2 = values[base + 2] & 0x3Fu;
          const uint32_t v3 = values[base + 3] & 0x3Fu;
          smem[k_major_mxf8f6f4_b_offset(col, byte_base + 0)] = pack_f6_byte0(v0, v1);
          smem[k_major_mxf8f6f4_b_offset(col, byte_base + 1)] = pack_f6_byte1(v1, v2);
          smem[k_major_mxf8f6f4_b_offset(col, byte_base + 2)] = pack_f6_byte2(v2, v3);
        }
      }
    } else if (type_code == 5) {
      for (int block = 0; block < 32; block += 16) {
        for (int pair = 0; pair < 8; ++pair) {
          const int base = block + pair * 2;
          const int byte_k = block + pair;
          smem[k_major_mxf8f6f4_b_offset(col, byte_k)] = pack_f4_pair(values[base + 0], values[base + 1]);
        }
      }
    } else {
      for (int k = 0; k < 32; ++k) {
        smem[k_major_mxf8f6f4_b_offset(col, k)] = mxf8f6f4_from_container(values[k]);
      }
    }
  }
}

__device__ __forceinline__ int k_major_fp4_a_element_offset(int row, int k) {
  // K-major no-swizzle canonical layout for 4-bit tcgen05 operands:
  // ((8,m),(T,2)):((1T,SBO),(1,LBO)), T=32, SBO=32 elements, LBO=64 elements.
  return (row & 7) * 32 + (row >> 3) * 32 + (k & 31) + (k >> 5) * 64;
}

__device__ __forceinline__ int k_major_fp4_b_element_offset(int col, int k) {
  return (col & 7) * 32 + (col >> 3) * 32 + (k & 31) + (k >> 5) * 64;
}

__device__ __forceinline__ void set_packed_fp4(uint8_t* smem, int element_offset, uint8_t value) {
  uint8_t& byte = smem[element_offset >> 1];
  if (element_offset & 1) {
    byte = static_cast<uint8_t>((byte & 0x0Fu) | (value << 4));
  } else {
    byte = static_cast<uint8_t>((byte & 0xF0u) | value);
  }
}

__device__ __forceinline__ void tmem_store_16x256(uint32_t taddr, uint32_t value) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %1, %1, %1};\n"
      :
      : "r"(taddr), "r"(value)
      : "memory");
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

__device__ __forceinline__ void issue_mxf8f6f4_1x(uint32_t tmem_base,
                                                 uint64_t desc_a,
                                                 uint64_t desc_b,
                                                 uint32_t idesc,
                                                 uint32_t scale_a_tmem,
                                                 uint32_t scale_b_tmem,
                                                 uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_mxf8f6f4_1x_cg2(uint32_t tmem_base,
                                                     uint64_t desc_a,
                                                     uint64_t desc_b,
                                                     uint32_t idesc,
                                                     uint32_t scale_a_tmem,
                                                     uint32_t scale_b_tmem,
                                                     uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_mxf4_2x(uint32_t tmem_base,
                                             uint64_t desc_a,
                                             uint64_t desc_b,
                                             uint32_t idesc,
                                             uint32_t scale_a_tmem,
                                             uint32_t scale_b_tmem,
                                             uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_mxf4_2x_cg2(uint32_t tmem_base,
                                                 uint64_t desc_a,
                                                 uint64_t desc_b,
                                                 uint32_t idesc,
                                                 uint32_t scale_a_tmem,
                                                 uint32_t scale_b_tmem,
                                                 uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_mxf4nvf4_2x(uint32_t tmem_base,
                                                 uint64_t desc_a,
                                                 uint64_t desc_b,
                                                 uint32_t idesc,
                                                 uint32_t scale_a_tmem,
                                                 uint32_t scale_b_tmem,
                                                 uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_mxf4nvf4_2x_cg2(uint32_t tmem_base,
                                                     uint64_t desc_a,
                                                     uint64_t desc_b,
                                                     uint32_t idesc,
                                                     uint32_t scale_a_tmem,
                                                     uint32_t scale_b_tmem,
                                                     uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_mxf4nvf4_4x(uint32_t tmem_base,
                                                 uint64_t desc_a,
                                                 uint64_t desc_b,
                                                 uint32_t idesc,
                                                 uint32_t scale_a_tmem,
                                                 uint32_t scale_b_tmem,
                                                 uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

__device__ __forceinline__ void issue_mxf4nvf4_4x_cg2(uint32_t tmem_base,
                                                     uint64_t desc_a,
                                                     uint64_t desc_b,
                                                     uint32_t idesc,
                                                     uint32_t scale_a_tmem,
                                                     uint32_t scale_b_tmem,
                                                     uint32_t enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}\n"
      :
      : "r"(tmem_base), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(scale_a_tmem),
        "r"(scale_b_tmem), "r"(enable_input_d)
      : "memory");
}

template <bool CtaGroup2>
__global__ __launch_bounds__(kThreads) void tcgen05_block_scaled_kernel(const ProbeCase* __restrict__ cases,
                                                                       uint32_t* __restrict__ outputs,
                                                                       size_t count,
                                                                       uint32_t op) {
  __shared__ SharedStorage shared;

  const size_t case_idx = CtaGroup2 ? static_cast<size_t>(blockIdx.x >> 1) : static_cast<size_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  uint32_t cluster_rank = 0;
  if constexpr (CtaGroup2) {
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cluster_rank));
  }
  const uint32_t pair_rank = cluster_rank & 1u;
  if (case_idx >= count || tid >= static_cast<int>(kThreads)) {
    return;
  }

  const ProbeCase pc = cases[case_idx];
  const uint32_t type_code = type_code_for_op(op);
  const uint32_t a_type_code = is_mxf8f6f4_op(op) ? pc.a_format : type_code;
  const uint32_t b_type_code = is_mxf8f6f4_op(op) ? pc.b_format : type_code;
  const int k_dim = k_for_op(op);
  const uint32_t n = pc.n;
  const uint32_t m = pc.m;

  if (tid == 0) {
    init_barrier(&shared.mma_barrier, 1);
  }
  __syncthreads();

  if (warp == 0) {
    const uint32_t dst = smem_u32(&shared.tmem_base);
    if constexpr (CtaGroup2) {
      asm volatile(
          "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
          :
          : "r"(dst), "r"(kTmemColumns)
          : "memory");
    } else {
      asm volatile(
          "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
          :
          : "r"(dst), "r"(kTmemColumns)
          : "memory");
    }
  }
  __syncthreads();
  if constexpr (CtaGroup2) {
    cluster_sync();
  }

  for (int i = tid; i < 512; i += kThreads) {
    shared.a_smem[i] = 0;
  }
  for (int i = tid; i < 256; i += kThreads) {
    shared.b_smem[i] = 0;
  }
  __syncthreads();

  if (tid == 0) {
    if (is_mxf8f6f4_op(op)) {
      fill_mxf8f6f4_a_operand(shared.a_smem, pc.a, a_type_code, 1);
      fill_mxf8f6f4_b_operand(shared.b_smem, pc.b, b_type_code, 1);
    } else {
      for (int row = 0; row < 1; ++row) {
        for (int k = 0; k < k_dim; ++k) {
          set_packed_fp4(shared.a_smem, k_major_fp4_a_element_offset(row, k), fp4_from_container(pc.a[k]));
        }
      }
      for (int col = 0; col < 1; ++col) {
        for (int k = 0; k < k_dim; ++k) {
          set_packed_fp4(shared.b_smem, k_major_fp4_b_element_offset(col, k), fp4_from_container(pc.b[k]));
        }
      }
    }
  }
  __syncthreads();
  proxy_async_fence_shared_cta();

  const uint32_t tbase = shared.tmem_base;
  const uint32_t warp_taddr = tbase + static_cast<uint32_t>(warp * 32);

  tmem_store_16x256(warp_taddr, pc.c);
  tmem_wait_st();

  const uint32_t scale_a_base = tbase + kScaleAOffset;
  const uint32_t scale_b_base = tbase + kScaleBOffset;
  tmem_store_16x256(scale_a_base + static_cast<uint32_t>(warp * 32), pack_scale_word(pc.scale_a));
  tmem_store_16x256(scale_b_base + static_cast<uint32_t>(warp * 32), pack_scale_word(pc.scale_b));
  tmem_wait_st();

  tcgen05_fence_before_sync();
  __syncthreads();
  if constexpr (CtaGroup2) {
    cluster_sync();
  }
  tcgen05_fence_after_sync();

  if ((!CtaGroup2 || pair_rank == 0) && tid == 0) {
    const uint64_t desc_a =
        is_mxf8f6f4_op(op) ? make_smem_desc(shared.a_smem, 64, 32) : make_smem_desc(shared.a_smem, 32, 16);
    const uint64_t desc_b =
        is_mxf8f6f4_op(op) ? make_smem_desc(shared.b_smem, 64, 32) : make_smem_desc(shared.b_smem, 32, 16);
    const uint32_t idesc =
        is_mxf8f6f4_op(op) ? make_mxf8f6f4_instr_desc(a_type_code, b_type_code, n, m)
                            : make_mxf4_instr_desc(n, m, !is_ue4m3_nvfp4_op(op));

    if (is_mxf8f6f4_op(op)) {
      if constexpr (CtaGroup2) {
        issue_mxf8f6f4_1x_cg2(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      } else {
        issue_mxf8f6f4_1x(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      }
    } else if (op == kOpTcgen05Mxf4E2M1Ue8m0M128N8K64) {
      if constexpr (CtaGroup2) {
        issue_mxf4_2x_cg2(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      } else {
        issue_mxf4_2x(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      }
    } else if (op == kOpTcgen05Mxf4Nvfp4E2M1Ue8m0M128N8K64Scale2X) {
      if constexpr (CtaGroup2) {
        issue_mxf4nvf4_2x_cg2(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      } else {
        issue_mxf4nvf4_2x(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      }
    } else if (op == kOpTcgen05Mxf4Nvfp4E2M1Ue4m3M128N8K64Scale4X) {
      if constexpr (CtaGroup2) {
        issue_mxf4nvf4_4x_cg2(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      } else {
        issue_mxf4nvf4_4x(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      }
    } else {
      if constexpr (CtaGroup2) {
        issue_mxf4nvf4_4x_cg2(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      } else {
        issue_mxf4nvf4_4x(tbase, desc_a, desc_b, idesc, scale_a_base, scale_b_base, 1);
      }
    }
    if constexpr (CtaGroup2) {
      tcgen05_commit_cg2(&shared.mma_barrier);
      wait_barrier(&shared.mma_barrier, 0);
    } else {
      tcgen05_commit(&shared.mma_barrier);
    }
  }

  if constexpr (!CtaGroup2) {
    wait_barrier(&shared.mma_barrier, 0);
  }
  tcgen05_fence_before_sync();
  if constexpr (CtaGroup2) {
    cluster_sync();
  }
  tcgen05_fence_after_sync();
  __syncthreads();

  uint32_t r0 = 0;
  uint32_t r1 = 0;
  uint32_t r2 = 0;
  uint32_t r3 = 0;
  tmem_load_16x256(warp_taddr, r0, r1, r2, r3);
  tmem_wait_ld();

  if ((!CtaGroup2 || pair_rank == 0) && tid == 0) {
    const size_t base = case_idx * kRegsPerThread;
    outputs[base + 0] = r0;
    outputs[base + 1] = r1;
    outputs[base + 2] = r2;
    outputs[base + 3] = r3;
  }

  __syncthreads();
  if constexpr (CtaGroup2) {
    cluster_sync();
  }
  if (warp == 0) {
    if constexpr (CtaGroup2) {
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::: "memory");
      asm volatile(
          "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
          :
          : "r"(tbase), "r"(kTmemColumns)
          : "memory");
    } else {
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
      asm volatile(
          "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
          :
          : "r"(tbase), "r"(kTmemColumns)
          : "memory");
    }
  }
}

std::vector<uint32_t> run_cases(const std::vector<ProbeCase>& cases,
                                int device,
                                uint32_t op,
                                uint32_t& lanes) {
  die_cuda(cudaSetDevice(device), "cudaSetDevice");

  ProbeCase* d_cases = nullptr;
  uint32_t* d_outputs = nullptr;
  const bool cta_group2 = !cases.empty() && cases.front().cta_group == 2;
  for (const ProbeCase& pc : cases) {
    if ((pc.cta_group == 2) != cta_group2) {
      throw std::runtime_error("cannot mix block-scaled cta_group::1 and cta_group::2 cases");
    }
  }
  const size_t case_bytes = sizeof(ProbeCase) * cases.size();
  lanes = 1;
  const size_t output_count = cases.size() * kRegsPerThread;
  const size_t output_bytes = sizeof(uint32_t) * output_count;

  if (!cases.empty()) {
    die_cuda(cudaMalloc(&d_cases, case_bytes), "cudaMalloc cases");
    die_cuda(cudaMalloc(&d_outputs, output_bytes), "cudaMalloc outputs");
    die_cuda(cudaMemcpy(d_cases, cases.data(), case_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy cases");

    if (cases.size() > static_cast<size_t>(0x7fffffff)) {
      throw std::runtime_error("too many cases for one launch");
    }
    if (cta_group2) {
      cudaLaunchAttribute attr{};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = 2;
      attr.val.clusterDim.y = 1;
      attr.val.clusterDim.z = 1;

      cudaLaunchConfig_t config{};
      config.gridDim = dim3(static_cast<unsigned int>(cases.size() * 2), 1, 1);
      config.blockDim = dim3(kThreads, 1, 1);
      config.dynamicSmemBytes = 0;
      config.stream = nullptr;
      config.attrs = &attr;
      config.numAttrs = 1;
      die_cuda(cudaLaunchKernelEx(&config, tcgen05_block_scaled_kernel<true>, d_cases, d_outputs,
                                  cases.size(), op),
               "kernel launch");
    } else {
      tcgen05_block_scaled_kernel<false><<<static_cast<unsigned int>(cases.size()), kThreads>>>(
          d_cases, d_outputs, cases.size(), op);
      die_cuda(cudaGetLastError(), "kernel launch");
    }
    die_cuda(cudaDeviceSynchronize(), "kernel sync");
  }

  std::vector<uint32_t> outputs(output_count);
  if (!outputs.empty()) {
    die_cuda(cudaMemcpy(outputs.data(), d_outputs, output_bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy outputs");
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
            << "\"op\":\"tcgen05 block-scaled mxf8f6f4/mxf4/mxf4nvf4 UE8M0\","
            << "\"shape\":\"m128n8k32/m128n8k64\","
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

    auto batch = read_cases(input_path);
    uint32_t lanes = kThreads;
    auto outputs = run_cases(batch.cases, device, batch.op, lanes);
    write_outputs(output_path, outputs, batch.cases.size(), batch.op, lanes);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "tcgen05_block_scaled_probe: " << ex.what() << "\n";
    return 1;
  }
}
