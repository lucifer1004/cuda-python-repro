#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[32];
  __shared__ uint64_t _mbarrier[4];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 8; ++k) {
      tl::mbarrier_wait(_mbarrier[1], ((k & 1) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[0], 8192);
        tl::tma_load(A_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[4096])), (k * 64), (((int)blockIdx.y) * 64));
        tl::mbarrier_expect_tx(_mbarrier[0], 8192);
        tl::tma_load(B_desc, _mbarrier[0], (&(((half_t*)buf_dyn_shmem)[0])), (k * 64), (((int)blockIdx.x) * 64));
      }
      tl::mbarrier_arrive(_mbarrier[0]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    tl::fence_proxy_async();
    for (int k_1 = 0; k_1 < 8; ++k_1) {
      tl::mbarrier_wait(_mbarrier[0], (k_1 & 1));
      tl::gemm_ss<64, 64, 64, 4, 1, 0, 1, 0, true>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[0])), (&(C_local[0])));
      tl::mbarrier_arrive(_mbarrier[1]);
    }
    tl::syncthreads_partial(_mbarrier[2]);
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i_1 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) + 4096)])), __pack_half2(((half_t)C_local[(i_1 * 8)]), ((half_t)C_local[((i_1 * 8) + 1)])), __pack_half2(((half_t)C_local[((i_1 * 8) + 2)]), ((half_t)C_local[((i_1 * 8) + 3)])), __pack_half2(((half_t)C_local[((i_1 * 8) + 4)]), ((half_t)C_local[((i_1 * 8) + 5)])), __pack_half2(((half_t)C_local[((i_1 * 8) + 6)]), ((half_t)C_local[((i_1 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    tl::syncthreads_partial(_mbarrier[3]);
    if (((int)threadIdx.x) == 0) {
      tl::tma_store(C_desc, (&(((half_t*)buf_dyn_shmem)[4096])), (((int)blockIdx.x) * 64), (((int)blockIdx.y) * 64));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}
