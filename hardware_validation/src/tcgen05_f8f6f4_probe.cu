// Binary harness for:
//   tcgen05.mma.cta_group::1.kind::f8f6f4 with E5M2/E2M3/E3M2/E2M1 A/B descriptor types
//
// Targeted instruction shape:
//   M=64, N=8, K=32, dense f8/f6/f4 x f8/f6/f4 -> F32 accumulator
//
// The kernel maps the same scalar dot-add
//   c + sum_{k=0..31} a[k] * b[k]
// onto every output element by filling every row of A and every column of B
// with the same K-vector. A/B input cases are uint32 containers; the runner
// stores the low 8 bits in shared memory. E2M3/E3M2 use the low 6 bits and
// E2M1 uses the low 4 bits of each byte container.

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
constexpr uint32_t kOpTcgen05F8f6f4E5M2M64N8K32 = 12;
constexpr uint32_t kOpTcgen05F8f6f4E2M3M64N8K32 = 13;
constexpr uint32_t kOpTcgen05F8f6f4E3M2M64N8K32 = 14;
constexpr uint32_t kOpTcgen05F8f6f4E2M1M64N8K32 = 15;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kRegsPerThread = 4;
constexpr uint32_t kTmemColumns = 32;
constexpr int kM = 64;
constexpr int kN = 8;
constexpr int kK = 32;

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
  uint32_t a[kK];
  uint32_t b[kK];
  uint32_t c;
};

struct InputBatch {
  uint32_t op = 0;
  std::vector<ProbeCase> cases;
};

struct alignas(128) SharedStorage {
  alignas(16) uint32_t tmem_base;
  alignas(8) uint64_t mma_barrier;
  alignas(128) uint8_t a_smem[512];
  alignas(128) uint8_t b_smem[256];
};

static_assert(sizeof(InputHeader) == 24, "unexpected InputHeader packing");
static_assert(sizeof(OutputHeader) == 32, "unexpected OutputHeader packing");
static_assert(sizeof(ProbeCase) == 260, "unexpected ProbeCase packing");

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

bool is_supported_op(uint32_t op) {
  return op == kOpTcgen05F8f6f4E5M2M64N8K32 || op == kOpTcgen05F8f6f4E2M3M64N8K32 ||
         op == kOpTcgen05F8f6f4E3M2M64N8K32 || op == kOpTcgen05F8f6f4E2M1M64N8K32;
}

uint32_t type_code_for_op(uint32_t op) {
  switch (op) {
    case kOpTcgen05F8f6f4E5M2M64N8K32:
      return 1;  // E5M2
    case kOpTcgen05F8f6f4E2M3M64N8K32:
      return 3;  // E2M3
    case kOpTcgen05F8f6f4E3M2M64N8K32:
      return 4;  // E3M2
    case kOpTcgen05F8f6f4E2M1M64N8K32:
      return 5;  // E2M1
    default:
      throw std::runtime_error("unsupported op id");
  }
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
  return batch;
}

void write_outputs(const std::string& path, const std::vector<uint32_t>& outputs, size_t count, uint32_t op) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }

  OutputHeader header{};
  std::memcpy(header.magic, kOutputMagic, sizeof(header.magic));
  header.version = kVersion;
  header.op = op;
  header.lanes = kThreads;
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

__device__ __forceinline__ uint32_t make_f8f6f4_instr_desc(uint32_t type_code) {
  uint32_t desc = 0;
  desc |= 1u << 4;       // D type: F32
  desc |= type_code << 7;
  desc |= type_code << 10;
  desc |= 1u << 17;      // N >> 3, N = 8
  desc |= 4u << 24;      // M >> 4, M = 64
  return desc;
}

__device__ __forceinline__ uint8_t f8f6f4_from_container(uint32_t bits) {
  return static_cast<uint8_t>(bits);
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

__device__ __forceinline__ int k_major_f8f6f4_a_offset(int row, int k) {
  // K-major no-swizzle canonical layout for 8-bit tcgen05 operands:
  // ((8,m),(T,2)):((1T,SBO),(1,LBO)), T=16, SBO=32 bytes, LBO=64 bytes.
  return (row & 7) * 16 + (row >> 3) * 32 + (k & 15) + (k >> 4) * 64;
}

__device__ __forceinline__ int k_major_f8f6f4_b_offset(int col, int k) {
  return (col & 7) * 16 + (col >> 3) * 32 + (k & 15) + (k >> 4) * 64;
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

__device__ __forceinline__ void issue_tcgen05_mma(uint32_t tmem_base,
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

__global__ __launch_bounds__(kThreads) void tcgen05_f8f6f4_kernel(const ProbeCase* __restrict__ cases,
                                                                 uint32_t* __restrict__ outputs,
                                                                 size_t count,
                                                                 uint32_t type_code) {
  __shared__ SharedStorage shared;

  const size_t case_idx = static_cast<size_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  if (case_idx >= count || tid >= static_cast<int>(kThreads)) {
    return;
  }

  const ProbeCase pc = cases[case_idx];

  if (tid == 0) {
    init_barrier(&shared.mma_barrier, 1);
  }
  __syncthreads();

  if (warp == 0) {
    const uint32_t dst = smem_u32(&shared.tmem_base);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(dst), "r"(kTmemColumns)
        : "memory");
  }
  __syncthreads();

  for (int i = tid; i < 512; i += kThreads) {
    shared.a_smem[i] = 0;
  }
  for (int i = tid; i < 256; i += kThreads) {
    shared.b_smem[i] = 0;
  }
  __syncthreads();

  for (int row = tid; row < kM; row += kThreads) {
    if (type_code == 3 || type_code == 4) {
      for (int block = 0; block < kK; block += 16) {
        for (int group = 0; group < 4; ++group) {
          const int base = block + group * 4;
          const int byte_base = block + group * 3;
          const uint32_t v0 = pc.a[base + 0] & 0x3fu;
          const uint32_t v1 = pc.a[base + 1] & 0x3fu;
          const uint32_t v2 = pc.a[base + 2] & 0x3fu;
          const uint32_t v3 = pc.a[base + 3] & 0x3fu;
          shared.a_smem[k_major_f8f6f4_a_offset(row, byte_base + 0)] = pack_f6_byte0(v0, v1);
          shared.a_smem[k_major_f8f6f4_a_offset(row, byte_base + 1)] = pack_f6_byte1(v1, v2);
          shared.a_smem[k_major_f8f6f4_a_offset(row, byte_base + 2)] = pack_f6_byte2(v2, v3);
        }
      }
    } else if (type_code == 5) {
      for (int block = 0; block < kK; block += 16) {
        for (int pair = 0; pair < 8; ++pair) {
          const int base = block + pair * 2;
          const int byte_k = block + pair;
          shared.a_smem[k_major_f8f6f4_a_offset(row, byte_k)] =
              pack_f4_pair(pc.a[base + 0], pc.a[base + 1]);
        }
      }
    } else {
      for (int k = 0; k < kK; ++k) {
        shared.a_smem[k_major_f8f6f4_a_offset(row, k)] = f8f6f4_from_container(pc.a[k]);
      }
    }
  }
  for (int col = tid; col < kN; col += kThreads) {
    if (type_code == 3 || type_code == 4) {
      for (int block = 0; block < kK; block += 16) {
        for (int group = 0; group < 4; ++group) {
          const int base = block + group * 4;
          const int byte_base = block + group * 3;
          const uint32_t v0 = pc.b[base + 0] & 0x3fu;
          const uint32_t v1 = pc.b[base + 1] & 0x3fu;
          const uint32_t v2 = pc.b[base + 2] & 0x3fu;
          const uint32_t v3 = pc.b[base + 3] & 0x3fu;
          shared.b_smem[k_major_f8f6f4_b_offset(col, byte_base + 0)] = pack_f6_byte0(v0, v1);
          shared.b_smem[k_major_f8f6f4_b_offset(col, byte_base + 1)] = pack_f6_byte1(v1, v2);
          shared.b_smem[k_major_f8f6f4_b_offset(col, byte_base + 2)] = pack_f6_byte2(v2, v3);
        }
      }
    } else if (type_code == 5) {
      for (int block = 0; block < kK; block += 16) {
        for (int pair = 0; pair < 8; ++pair) {
          const int base = block + pair * 2;
          const int byte_k = block + pair;
          shared.b_smem[k_major_f8f6f4_b_offset(col, byte_k)] =
              pack_f4_pair(pc.b[base + 0], pc.b[base + 1]);
        }
      }
    } else {
      for (int k = 0; k < kK; ++k) {
        shared.b_smem[k_major_f8f6f4_b_offset(col, k)] = f8f6f4_from_container(pc.b[k]);
      }
    }
  }
  __syncthreads();
  proxy_async_fence_shared_cta();

  const uint32_t tbase = shared.tmem_base;
  const uint32_t warp_taddr = tbase + static_cast<uint32_t>(warp * 32);

  tmem_store_16x256(warp_taddr, pc.c);
  tmem_wait_st();
  tcgen05_fence_before_sync();
  __syncthreads();
  tcgen05_fence_after_sync();

  if (tid == 0) {
    const uint64_t desc_a = make_smem_desc(shared.a_smem, 64 * sizeof(uint8_t), 32 * sizeof(uint8_t));
    const uint64_t desc_b = make_smem_desc(shared.b_smem, 64 * sizeof(uint8_t), 32 * sizeof(uint8_t));
    issue_tcgen05_mma(tbase, desc_a, desc_b, make_f8f6f4_instr_desc(type_code), 1);
    tcgen05_commit(&shared.mma_barrier);
  }

  wait_barrier(&shared.mma_barrier, 0);
  tcgen05_fence_after_sync();
  __syncthreads();

  uint32_t r0 = 0;
  uint32_t r1 = 0;
  uint32_t r2 = 0;
  uint32_t r3 = 0;
  tmem_load_16x256(warp_taddr, r0, r1, r2, r3);
  tmem_wait_ld();

  const size_t base = ((case_idx * kThreads) + static_cast<size_t>(tid)) * kRegsPerThread;
  outputs[base + 0] = r0;
  outputs[base + 1] = r1;
  outputs[base + 2] = r2;
  outputs[base + 3] = r3;

  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
        :
        : "r"(tbase), "r"(kTmemColumns)
        : "memory");
  }
}

std::vector<uint32_t> run_cases(const std::vector<ProbeCase>& cases, int device, uint32_t op) {
  die_cuda(cudaSetDevice(device), "cudaSetDevice");
  const uint32_t type_code = type_code_for_op(op);

  ProbeCase* d_cases = nullptr;
  uint32_t* d_outputs = nullptr;
  const size_t case_bytes = sizeof(ProbeCase) * cases.size();
  const size_t output_count = cases.size() * kThreads * kRegsPerThread;
  const size_t output_bytes = sizeof(uint32_t) * output_count;

  if (!cases.empty()) {
    die_cuda(cudaMalloc(&d_cases, case_bytes), "cudaMalloc cases");
    die_cuda(cudaMalloc(&d_outputs, output_bytes), "cudaMalloc outputs");
    die_cuda(cudaMemcpy(d_cases, cases.data(), case_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy cases");

    if (cases.size() > static_cast<size_t>(0x7fffffff)) {
      throw std::runtime_error("too many cases for one launch");
    }
    tcgen05_f8f6f4_kernel<<<static_cast<unsigned int>(cases.size()), kThreads>>>(
        d_cases, d_outputs, cases.size(), type_code);
    die_cuda(cudaGetLastError(), "kernel launch");
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
            << "\"compute_capability\":\"" << prop.major << "." << prop.minor << "\","
            << "\"op\":\"tcgen05.mma.cta_group::1.kind::f8f6f4.e5m2/e2m3/e3m2/e2m1\","
            << "\"shape\":\"m64n8k32\","
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
    auto outputs = run_cases(batch.cases, device, batch.op);
    write_outputs(output_path, outputs, batch.cases.size(), batch.op);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "tcgen05_f8f6f4_probe: " << ex.what() << "\n";
    return 1;
  }
}
