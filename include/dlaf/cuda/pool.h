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
#include "dlaf/cuda/error.h"

namespace dlaf {

// Print information about the devices.
inline void print_device_info(int device) noexcept {
  cudaDeviceProp device_prop;
  DLAF_CUDA_CALL(cudaGetDeviceProperties(&device_prop, device));
  printf("Device %d has compute capability %d.%d.\n", device, device_prop.major, device_prop.minor);
}

// Return the number of devices.
inline int num_devices() noexcept {
  int ndevices;
  cudaGetDeviceCount(&ndevices);
  return ndevices;
}

// For multi-threaded applications that use the same device from different threads, the recommended
// programming model is to create one CUBLAS handle per thread and use that CUBLAS handle for the entire
// life of the thread. [1]
//
// [1]: cuBLAS Guide, 2.4 cuBLAS Helper Function Reference, cublasCreate()
//

// For static frameworks the recommended best practice is to create all the needed streams upfront and
// destroy them after the work is done.
//
// https://github.com/pytorch/pytorch/issues/9646
// TODO: Note: Streams are thread safe.
//
// https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDAStream.h
// Create a pool for cublas_handles associated to different devices and streams.
//
// TODO: clarify implicit synchronizations ?
// TODO: clarify Streams are a FIFO structure [1].
//
// [1]: CUDA Programming Guide, 3.2.5.5.4. Implicit Synchronization
//
// TODO: The pool is not thread safe!
//
class cublas_pool {
  int device_;
  std::vector<cublasHandle_t> handles_arr_;
  int curr_handle_id_;

public:
  cublas_pool(cublas_pool&&) = default;
  cublas_pool& operator=(cublas_pool&&) = default;
  cublas_pool(const cublas_pool&) = delete;
  cublas_pool& operator=(const cublas_pool&) = delete;

  /// The number of streams spawned per device depends on the supported maximum number of concurrent
  /// kernels that can execute in parallel on that device [1]. The number varies between different GPU
  /// models, most support at least 16.
  ///
  /// [1]: CUDA Programming Guide, Table 15. Technical Specifications per Compute Capability
  cublas_pool(int device, int num_streams) noexcept : device_{device}, curr_handle_id_{0} {
    DLAF_ASSERT(num_streams >= 1, "At least 1 stream needs to be specified!");
    DLAF_ASSERT(device >= 0, "Valid device IDs start from 0!");
    DLAF_ASSERT(device < num_devices(), "The device ID exceeds the number of available devices!");

    handles_arr_.reserve(static_cast<std::size_t>(num_streams));

    // - A kernel launch will fail if it is issued to a stream that is not associated to the current
    //   device. [1]
    //
    // [1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
    cudaSetDevice(device);
    for (int i_stream = 0; i_stream < num_streams; ++i_stream) {
      cudaStream_t stream;
      DLAF_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      cublasHandle_t handle;
      DLAF_CUBLAS_CALL(cublasCreate(&handle));
      DLAF_CUBLAS_CALL(cublasSetStream(handle, stream));
      handles_arr_.push_back(handle);
    }
  }

  ~cublas_pool() noexcept {
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

  /// Return the number of cuBLAS handles available to this device.
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
  cublas_pool& set_handle_id(int handle_id) noexcept {
    DLAF_ASSERT(handle_id >= 0);
    DLAF_ASSERT(handle_id < num_handles());
    curr_handle_id_ = handle_id;
    return *this;
  }
};

}
