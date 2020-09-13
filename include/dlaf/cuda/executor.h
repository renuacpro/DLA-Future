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
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <hpx/include/async.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/util/yield_while.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/event.h"
#include "dlaf/cuda/mutex.h"
#include "dlaf/cuda/pool.h"

namespace dlaf {
namespace cuda {

/// An executor for CUDA calls.
struct executor {
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  inline executor(cuda::pool& pool, bool pointer_mode_host)
      : device_(pool.device_id()), stream_(pool.stream()),
        threads_executor_("default", hpx::threads::thread_priority_high) {}

  constexpr bool operator==(executor const& rhs) const noexcept {
    return device_ == rhs.device_ && stream_ == rhs.stream_ &&
           threads_executor_ == rhs.threads_executor_;
  }

  constexpr bool operator!=(executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  constexpr executor const& context() const noexcept {
    return *this;
  }

  // Implement the TwoWayExecutor interface.
  //
  // Note: the member can't be marked `const` because of `threads_executor_`.
  // Note: Parameters are passed by value as they are small types: pointers, integers or scalars.
  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F f, Ts... ts) {
    // Set the device corresponding to the CUDA stream.
    //
    // The CUBLAS library context is tied to the current CUDA device [1]. A previous task scheduled on
    // the same thread may have set a different device, this makes sure the correct device is used. The
    // function is considered very low overhead call [2]. Any previous assignment of CUDA devices to
    // threads is not preserved.
    //
    // [1]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
    // [2]: CUDA Runtime API, section 5.1 Device Management
    DLAF_CUDA_CALL(cudaSetDevice(device_));

    // Use an event to query the CUBLAS kernel for completion. Timing is disabled for performance. [1]
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
      DLAF_CUDA_CALL(f(ts..., stream_));
      ev.record(stream_);
    }
    return hpx::async(threads_executor_, [e = std::move(ev)] { e.query(); });
  }

private:
  int device_;
  cudaStream_t stream_;
  hpx::threads::executors::pool_executor threads_executor_;
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
