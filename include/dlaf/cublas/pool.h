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

#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "dlaf/common/assert.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cuda/pool.h"

namespace dlaf {
namespace cublas {

// A pool of handles and streams assigned to a `device`.
//
// To amortize creation/destruction costs with CUBLAS handles/streams, the recommended best practice is
// to create all the needed CUBLAS handles/streams upfront and destroy them after the work is done [1].
//
// [1]: https://github.com/pytorch/pytorch/issues/9646
class pool {
  int device_;
  std::vector<cublasHandle_t> handles_arr_;
  int curr_handle_id_;

public:
  pool(pool&&) = default;
  pool& operator=(pool&&) = default;
  pool(const pool&) = delete;
  pool& operator=(const pool&) = delete;

  /// The number of streams spawned per device depends on the supported maximum number of concurrent
  /// kernels that can execute in parallel on that device [1]. The number varies between different GPU
  /// models, most support at least 16.
  ///
  /// [1]: CUDA Programming Guide, Table 15. Technical Specifications per Compute Capability
  pool(const cuda::pool& cuda_pool) noexcept : device_{cuda_pool.device_id()}, curr_handle_id_{0} {
    handles_arr_.reserve(static_cast<std::size_t>(cuda_pool.num_streams()));

    // A kernel launch will fail if it is issued to a stream that is not associated to the current device. [1]
    //
    // [1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
    cudaSetDevice(device_);
    for (cudaStream_t stream : cuda_pool.streams_arr()) {
      cublasHandle_t handle;
      DLAF_CUBLAS_CALL(cublasCreate(&handle));
      DLAF_CUBLAS_CALL(cublasSetStream(handle, stream));
      handles_arr_.push_back(handle);
    }
  }

  ~pool() noexcept {
    for (cublasHandle_t handle : handles_arr_) {
      // This implicitly calls `cublasDeviceSynchronize()` [1].
      //
      // [1]: cuBLAS, section 2.4 cuBLAS Helper Function Reference
      DLAF_CUBLAS_CALL(cublasDestroy(handle));
    }
  }

  /// Return the device ID of the pool.
  int device_id() const noexcept {
    return device_;
  }

  /// Return the number of cuBLAS handles available in the pool.
  int num_handles() const noexcept {
    return static_cast<int>(handles_arr_.size());
  }

  /// Return the current handle ID.
  int current_handle_id() const noexcept {
    return curr_handle_id_;
  }

  /// Return handles in Round-Robin.
  cublasHandle_t handle() noexcept {
    cublasHandle_t hd = handles_arr_[static_cast<std::size_t>(curr_handle_id_)];
    ++curr_handle_id_;
    if (curr_handle_id_ == num_handles())
      curr_handle_id_ = 0;
    return hd;
  }

  /// Set the current `handle_id`.
  ///
  /// This resets the Round-Robin process starting from `handle_id`.
  pool& set_handle_id(int handle_id) noexcept {
    DLAF_ASSERT(handle_id >= 0, "");
    DLAF_ASSERT(handle_id < num_handles(), "");
    curr_handle_id_ = handle_id;
    return *this;
  }

  std::vector<cublasHandle_t> const& handles_arr() const noexcept {
    return handles_arr_;
  }
};
}
}
