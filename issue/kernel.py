
import ctypes

import cuda.bindings.driver

_function_names = ['main_kernel']


def call(kernels, A, B, C, stream=0):

    A_desc_type = cuda.bindings.driver.CUtensorMapDataType(6)
    A_desc_tensorRank = 2
    A_desc_globalAddress = A.data_ptr()
    A_desc_globalDim = [cuda.bindings.driver.cuuint64_t(
        512), cuda.bindings.driver.cuuint64_t(512)]
    A_desc_globalStride = [cuda.bindings.driver.cuuint64_t(
        2), cuda.bindings.driver.cuuint64_t(1024)][1:]
    A_desc_boxDim = [cuda.bindings.driver.cuuint32_t(
        64), cuda.bindings.driver.cuuint32_t(64)]
    A_desc_elementStrides = [cuda.bindings.driver.cuuint32_t(
        1), cuda.bindings.driver.cuuint32_t(1)]
    A_desc_interleave = cuda.bindings.driver.CUtensorMapInterleave(0)
    A_desc_swizzle = cuda.bindings.driver.CUtensorMapSwizzle(3)
    A_desc_l2Promotion = cuda.bindings.driver.CUtensorMapL2promotion(2)
    A_desc_oobFill = cuda.bindings.driver.CUtensorMapFloatOOBfill(0)

    res, A_desc = cuda.bindings.driver.cuTensorMapEncodeTiled(
        A_desc_type,
        A_desc_tensorRank,
        A_desc_globalAddress,
        A_desc_globalDim,
        A_desc_globalStride,
        A_desc_boxDim,
        A_desc_elementStrides,
        A_desc_interleave,
        A_desc_swizzle,
        A_desc_l2Promotion,
        A_desc_oobFill,
    )

    if res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to initialize the TMA descriptor A_desc: {res}")

    B_desc_type = cuda.bindings.driver.CUtensorMapDataType(6)
    B_desc_tensorRank = 2
    B_desc_globalAddress = B.data_ptr()
    B_desc_globalDim = [cuda.bindings.driver.cuuint64_t(
        512), cuda.bindings.driver.cuuint64_t(512)]
    B_desc_globalStride = [cuda.bindings.driver.cuuint64_t(
        2), cuda.bindings.driver.cuuint64_t(1024)][1:]
    B_desc_boxDim = [cuda.bindings.driver.cuuint32_t(
        64), cuda.bindings.driver.cuuint32_t(64)]
    B_desc_elementStrides = [cuda.bindings.driver.cuuint32_t(
        1), cuda.bindings.driver.cuuint32_t(1)]
    B_desc_interleave = cuda.bindings.driver.CUtensorMapInterleave(0)
    B_desc_swizzle = cuda.bindings.driver.CUtensorMapSwizzle(3)
    B_desc_l2Promotion = cuda.bindings.driver.CUtensorMapL2promotion(2)
    B_desc_oobFill = cuda.bindings.driver.CUtensorMapFloatOOBfill(0)

    res, B_desc = cuda.bindings.driver.cuTensorMapEncodeTiled(
        B_desc_type,
        B_desc_tensorRank,
        B_desc_globalAddress,
        B_desc_globalDim,
        B_desc_globalStride,
        B_desc_boxDim,
        B_desc_elementStrides,
        B_desc_interleave,
        B_desc_swizzle,
        B_desc_l2Promotion,
        B_desc_oobFill,
    )

    if res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to initialize the TMA descriptor B_desc: {res}")

    C_desc_type = cuda.bindings.driver.CUtensorMapDataType(6)
    C_desc_tensorRank = 2
    C_desc_globalAddress = C.data_ptr()
    C_desc_globalDim = [cuda.bindings.driver.cuuint64_t(
        512), cuda.bindings.driver.cuuint64_t(512)]
    C_desc_globalStride = [cuda.bindings.driver.cuuint64_t(
        2), cuda.bindings.driver.cuuint64_t(1024)][1:]
    C_desc_boxDim = [cuda.bindings.driver.cuuint32_t(
        64), cuda.bindings.driver.cuuint32_t(64)]
    C_desc_elementStrides = [cuda.bindings.driver.cuuint32_t(
        1), cuda.bindings.driver.cuuint32_t(1)]
    C_desc_interleave = cuda.bindings.driver.CUtensorMapInterleave(0)
    C_desc_swizzle = cuda.bindings.driver.CUtensorMapSwizzle(0)
    C_desc_l2Promotion = cuda.bindings.driver.CUtensorMapL2promotion(2)
    C_desc_oobFill = cuda.bindings.driver.CUtensorMapFloatOOBfill(0)

    res, C_desc = cuda.bindings.driver.cuTensorMapEncodeTiled(
        C_desc_type,
        C_desc_tensorRank,
        C_desc_globalAddress,
        C_desc_globalDim,
        C_desc_globalStride,
        C_desc_boxDim,
        C_desc_elementStrides,
        C_desc_interleave,
        C_desc_swizzle,
        C_desc_l2Promotion,
        C_desc_oobFill,
    )

    if res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to initialize the TMA descriptor C_desc: {res}")

    res = cuda.bindings.driver.cuKernelSetAttribute(
        cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        16384,
        kernels["main_kernel"],
        cuda.bindings.driver.CUdevice(0)
    )[0]
    if res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to set max dynamic shared memory size to 16384 for kernel main_kernel: {res}")

    config = cuda.bindings.driver.CUlaunchConfig()
    config.gridDimX = 8
    config.gridDimY = 8
    config.gridDimZ = 1
    config.blockDimX = 256
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 16384
    config.hStream = stream

    arg_values = A_desc, B_desc, C_desc
    arg_types = None, None, None

    res = cuda.bindings.driver.cuLaunchKernelEx(
        config, kernels["main_kernel"], (arg_values, arg_types), 0)[0]
    if res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel main_kernel: {res}")
