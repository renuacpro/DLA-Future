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

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/modules/execution_base.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/event.h"
#include "dlaf/cuda/mutex.h"

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

namespace internal {

struct cuda_stream_wrapper {
  cudaStream_t stream;
  cuda_stream_wrapper() = default;
  cuda_stream_wrapper(int device) noexcept {
    DLAF_CUDA_CALL(cudaSetDevice(device));
    DLAF_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  ~cuda_stream_wrapper() {
    DLAF_CUDA_CALL(cudaStreamDestroy(stream));
  }
};

}

/// An executor for CUDA calls.
///
/// Note: The streams are rotated in Round-robin.
class executor {
  using streams_arr_t = std::vector<internal::cuda_stream_wrapper>;

protected:
  int device_;
  std::shared_ptr<streams_arr_t> streams_ptr_;
  hpx::threads::executors::pool_executor threads_executor_;
  int curr_stream_idx_;

public:
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  executor(int device, int num_streams)
      : device_(device), streams_ptr_(std::make_shared<streams_arr_t>(num_streams, device)),
        threads_executor_("default", hpx::threads::thread_priority_high) {}

  bool operator==(executor const& rhs) const noexcept {
    return streams_ptr_ == rhs.streams_ptr_ && threads_executor_ == rhs.threads_executor_;
  }

  bool operator!=(executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  executor const& context() const noexcept {
    return *this;
  }

  executor& set_curr_stream_idx(int curr_stream_idx) noexcept {
    DLAF_ASSERT(curr_stream_idx >= 0, curr_stream_idx);
    DLAF_ASSERT(curr_stream_idx < streams_ptr_->size(), curr_stream_idx);
    curr_stream_idx_ = curr_stream_idx;
    return *this;
  }

  // Implement the TwoWayExecutor interface.
  //
  // Note: the member can't be `const` because of `threads_executor_`.
  // Note: Parameters are passed by value as they are small types: pointers, integers or scalars.
  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F f, Ts... ts) {
    // Set the device to associate the following event (`ev`) with it. [1]
    //
    // A previous task scheduled on the same thread may have set a different device, this makes sure the
    // correct device is used. The function is considered very low overhead [2]. Any previous assignment
    // of CUDA devices to threads is not preserved.
    //
    // [1]: CUDA Programming Guide, section 3.2.7.2 Device selection
    // [2]: CUDA Runtime API, section 5.1 Device Management
    DLAF_CUDA_CALL(cudaSetDevice(device_));

    // Use an event to query the CUDA kernel for completion. Timing is disabled for performance. [1]
    //
    // [1]: CUDA Runtime API, section 5.5 Event Management
    cuda::Event ev{};

    // Call the CUDA function `f` and schedule an event after it.
    //
    // The event indicates the the function `f` has completed. The stream may be shared by mutliple
    // host threads, the mutex is here to make sure no other CUDA calls or events are scheduled
    // between the call to `f` and it's corresponding event.
    {
      std::lock_guard<hpx::lcos::local::mutex> lk(internal::get_cuda_mtx());
      cudaStream_t stream = (*streams_ptr_)[curr_stream_idx_].stream;
      DLAF_CUDA_CALL(f(ts..., stream));
      ev.record(stream);
      curr_stream_idx_ = (curr_stream_idx_ + 1) % streams_ptr_->size();
    }
    return hpx::async(threads_executor_, [e = std::move(ev)] { e.query(); });
  }
};

}
}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::cuda::executor> : std::true_type {};

}
}
}
