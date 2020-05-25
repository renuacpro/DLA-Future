//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <exception>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "dlaf/common/utils.h"

namespace dlaf {
namespace internal {

inline void cuda_call(cudaError_t err, const dlaf::common::internal::source_location& info) noexcept {
  if (err != cudaSuccess) {
    std::cout << "[CUDA ERROR] " << info << std::endl << cudaGetErrorString(err) << std::endl;
    std::terminate();
  }
}

#define DLAF_CUDA_CALL(cuda_f) dlaf::internal::cuda_call((cuda_f), SOURCE_LOCATION())

/// CUBLAS equivalent to `cudaGetErrorString()`
/// Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t
inline std::string cublas_get_err_string(cublasStatus_t st) {
  // clang-format off
  switch (st) {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  // clang-format on
  return "UNKNOWN";
}

inline void cublas_call(cublasStatus_t st,
                        const dlaf::common::internal::source_location& info) noexcept {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::cout << "[CUBLAS ERROR] " << info << std::endl << cublas_get_err_string(st) << std::endl;
    std::terminate();
  }
}

#define DLAF_CUBLAS_CALL(cublas_f) dlaf::internal::cublas_call((cublas_f), SOURCE_LOCATION())

}
}
