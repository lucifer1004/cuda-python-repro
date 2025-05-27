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


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 16384, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap A_desc;
	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)6;
	cuuint32_t A_desc_tensorRank= 2;
	void *A_desc_globalAddress= A;
	cuuint64_t A_desc_globalDim[2]= {512,512};
	cuuint64_t A_desc_globalStride[2]= {2,1024};
	cuuint32_t A_desc_boxDim[2]= {64,64};
	cuuint32_t A_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult A_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);

	if (A_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor A_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap B_desc;
	CUtensorMapDataType B_desc_type= (CUtensorMapDataType)6;
	cuuint32_t B_desc_tensorRank= 2;
	void *B_desc_globalAddress= B;
	cuuint64_t B_desc_globalDim[2]= {512,512};
	cuuint64_t B_desc_globalStride[2]= {2,1024};
	cuuint32_t B_desc_boxDim[2]= {64,64};
	cuuint32_t B_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion B_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill B_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult B_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);

	if (B_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor B_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap C_desc;
	CUtensorMapDataType C_desc_type= (CUtensorMapDataType)6;
	cuuint32_t C_desc_tensorRank= 2;
	void *C_desc_globalAddress= C;
	cuuint64_t C_desc_globalDim[2]= {512,512};
	cuuint64_t C_desc_globalStride[2]= {2,1024};
	cuuint32_t C_desc_boxDim[2]= {64,64};
	cuuint32_t C_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave C_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle C_desc_swizzle= (CUtensorMapSwizzle)0;
	CUtensorMapL2promotion C_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill C_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult C_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &C_desc, C_desc_type, C_desc_tensorRank, C_desc_globalAddress, C_desc_globalDim, C_desc_globalStride + 1, C_desc_boxDim, C_desc_elementStrides, C_desc_interleave, C_desc_swizzle, C_desc_l2Promotion, C_desc_oobFill);

	if (C_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor C_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	main_kernel<<<dim3(8, 8, 1), dim3(256, 1, 1), 16384, stream>>>(A_desc, B_desc, C_desc);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
