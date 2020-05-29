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
namespace cuda {

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

// A pool of streams assigned to a `device`.
//
// To amortize creation/destruction costs with CUDA streams, the recommended best practice is
// to create all the needed CUDA streams upfront and destroy them after the work is done [1].
//
// [1]: https://github.com/pytorch/pytorch/issues/9646
class pool {
  int device_;
  std::vector<cudaStream_t> streams_arr_;
  int curr_stream_id_;

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
  pool(int device, int num_streams) noexcept : device_{device}, curr_stream_id_{0} {
    DLAF_ASSERT(num_streams >= 1, "At least 1 stream needs to be specified!");
    DLAF_ASSERT(device >= 0, "Valid device IDs start from 0!");
    DLAF_ASSERT(device < num_devices(), "The device ID exceeds the number of available devices!");

    streams_arr_.reserve(static_cast<std::size_t>(num_streams));

    // - A kernel launch will fail if it is issued to a stream that is not associated to the current
    //   device. [1]
    //
    // [1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
    cudaSetDevice(device);
    for (int i_stream = 0; i_stream < num_streams; ++i_stream) {
      cudaStream_t stream;
      DLAF_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      streams_arr_.push_back(stream);
    }
  }

  ~pool() noexcept {
    for (cudaStream_t stream : streams_arr_) {
      DLAF_CUDA_CALL(cudaStreamDestroy(stream));
    }
  }

  /// Return the device ID of the pool.
  int device_id() const noexcept {
    return device_;
  }

  /// Return the number of streams available to this device in the pool.
  int num_streams() const noexcept {
    return static_cast<int>(streams_arr_.size());
  }

  /// Return the current stream ID.
  int current_stream_id() const noexcept {
    return curr_stream_id_;
  }

  /// Return streams in Round-Robin.
  cudaStream_t stream() noexcept {
    cudaStream_t stream = streams_arr_[static_cast<std::size_t>(curr_stream_id_)];
    ++curr_stream_id_;
    if (curr_stream_id_ == num_streams())
      curr_stream_id_ = 0;
    return stream;
  }

  /// Set the current `stream_id`.
  ///
  /// This resets the Round-Robin process starting from `stream_id`.
  pool& set_stream_id(int stream_id) noexcept {
    DLAF_ASSERT(stream_id >= 0);
    DLAF_ASSERT(stream_id < num_streams());
    curr_stream_id_ = stream_id;
    return *this;
  }

  std::vector<cudaStream_t> const& streams_arr() const noexcept {
    return streams_arr_;
  }
};
}
}
