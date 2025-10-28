#include <cuda_fp16.h>
#include <cuda.h>

// Define half_t if not already defined
#ifndef half_t
using half_t = __half;
#endif

// CUTLASS CUDA driver wrapper macro
#ifndef CUTLASS_CUDA_DRIVER_WRAPPER_CALL
#define CUTLASS_CUDA_DRIVER_WRAPPER_CALL(func) func
#endif

extern "C" void benchmark_tma_creation(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, int num_iterations) {
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Create A descriptor
        CUtensorMap A_desc;
        CUtensorMapDataType A_desc_type = (CUtensorMapDataType)6;
        cuuint32_t A_desc_tensorRank = 2;
        void *A_desc_globalAddress = A;
        cuuint64_t A_desc_globalDim[2] = {512, 512};
        cuuint64_t A_desc_globalStride[2] = {2, 1024};
        cuuint32_t A_desc_boxDim[2] = {64, 64};
        cuuint32_t A_desc_elementStrides[2] = {1, 1};
        CUtensorMapInterleave A_desc_interleave = (CUtensorMapInterleave)0;
        CUtensorMapSwizzle A_desc_swizzle = (CUtensorMapSwizzle)3;
        CUtensorMapL2promotion A_desc_l2Promotion = (CUtensorMapL2promotion)2;
        CUtensorMapFloatOOBfill A_desc_oobFill = (CUtensorMapFloatOOBfill)0;
        
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, 
            A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, 
            A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);
        
        // Create B descriptor
        CUtensorMap B_desc;
        CUtensorMapDataType B_desc_type = (CUtensorMapDataType)6;
        cuuint32_t B_desc_tensorRank = 2;
        void *B_desc_globalAddress = B;
        cuuint64_t B_desc_globalDim[2] = {512, 512};
        cuuint64_t B_desc_globalStride[2] = {2, 1024};
        cuuint32_t B_desc_boxDim[2] = {64, 64};
        cuuint32_t B_desc_elementStrides[2] = {1, 1};
        CUtensorMapInterleave B_desc_interleave = (CUtensorMapInterleave)0;
        CUtensorMapSwizzle B_desc_swizzle = (CUtensorMapSwizzle)3;
        CUtensorMapL2promotion B_desc_l2Promotion = (CUtensorMapL2promotion)2;
        CUtensorMapFloatOOBfill B_desc_oobFill = (CUtensorMapFloatOOBfill)0;
        
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, 
            B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, 
            B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);
        
        // Create C descriptor
        CUtensorMap C_desc;
        CUtensorMapDataType C_desc_type = (CUtensorMapDataType)6;
        cuuint32_t C_desc_tensorRank = 2;
        void *C_desc_globalAddress = C;
        cuuint64_t C_desc_globalDim[2] = {512, 512};
        cuuint64_t C_desc_globalStride[2] = {2, 1024};
        cuuint32_t C_desc_boxDim[2] = {64, 64};
        cuuint32_t C_desc_elementStrides[2] = {1, 1};
        CUtensorMapInterleave C_desc_interleave = (CUtensorMapInterleave)0;
        CUtensorMapSwizzle C_desc_swizzle = (CUtensorMapSwizzle)0;
        CUtensorMapL2promotion C_desc_l2Promotion = (CUtensorMapL2promotion)2;
        CUtensorMapFloatOOBfill C_desc_oobFill = (CUtensorMapFloatOOBfill)0;
        
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &C_desc, C_desc_type, C_desc_tensorRank, C_desc_globalAddress, C_desc_globalDim, 
            C_desc_globalStride + 1, C_desc_boxDim, C_desc_elementStrides, C_desc_interleave, 
            C_desc_swizzle, C_desc_l2Promotion, C_desc_oobFill);
    }
}
