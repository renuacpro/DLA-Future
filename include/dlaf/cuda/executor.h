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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/util/yield_while.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/pool.h"

namespace dlaf {

namespace internal {

// Call the CUBLAS function and schedule an event after it to query when it's done.
//
// The function calls `cudaSetDevice()` multiple times on each thread, any previous assignment of CUDA
// devices to threads is not preserved.
//
// Note: It is not recommended that multiple thread share the same CUBLAS handle because extreme care
//       needs to be taken when changing or destroying the handle.
template <typename F, typename... Ts>
void run_cublas(int device, cublasHandle_t handle, F&& f, Ts&&... ts) noexcept {
  // Set the device corresponding to the CUBLAS handle.
  //
  // The CUBLAS library context is tied to the current CUDA device [1]. A previous task scheduled on the
  // same thread may have set a different device, this makes sure the correct device is used. The
  // function is considered very low overhead call [2].
  //
  // [1]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
  // [2]: CUDA Runtime API, section 5.1 Device Management
  DLAF_CUDA_CALL(cudaSetDevice(device));

  // Use an event to query the CUBLAS kernel for completion. Timing is disabled for performance. [1]
  //
  // [1]: CUDA Runtime API, section 5.5 Event Management
  cudaEvent_t event;
  DLAF_CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

  // Get the stream on which the CUBLAS call is to execute.
  cudaStream_t stream;
  DLAF_CUBLAS_CALL(cublasGetStream(handle, &stream));

  // Call the CUBLAS function `f` and schedule an event after it.
  //
  // The event indicates the the function `f` has completed. The handle may be shared by mutliple host
  // threads, the mutex is here to make sure no other CUBLAS calls or events are scheduled between the
  // call to `f` and it's corresponding event.
  {
    static hpx::lcos::local::mutex mt;
    std::lock_guard<hpx::lcos::local::mutex> lk(mt);
    // TODO: set pointer mode?
    DLAF_CUBLAS_CALL(f(handle, std::forward<Ts>(ts)...));
    DLAF_CUDA_CALL(cudaEventRecord(event, stream));
  }
  hpx::util::yield_while([event] {
    // Note that the call succeeds even if the event is associated to a device that is different from the
    // current device on the host thread. That could be the case if a previous task executing on the same
    // host thread set a different device. [1]
    //
    // [1]: CUDA Programming Guide, section 3.2.3.6, Stream and Event Behavior
    cudaError_t err = cudaEventQuery(event);

    // Note: Despite the name, `cudaErrorNotReady` is not considered an error. [1]
    //
    // [1]: CUDA Runtime API, section 5.33 Data types used by CUDA Runtime
    if (err == cudaErrorNotReady) {
      return true;
    }
    else if (err == cudaSuccess) {
      return false;
    }
    DLAF_CUDA_CALL(err);  // An error occured, report and terminate.
    return false;         // Unreachable!
  });
  DLAF_CUDA_CALL(cudaEventDestroy(event));
}

}

// An executor for CUBLAS functions.
struct cublas_executor {
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  // The pool of host threads on which the calls to handle and device will be made.
  cublas_executor(cublas_pool& pool, hpx::threads::executors::pool_executor threads_executor)
      : device_(pool.device_id()), handle_(pool.handle()),
        threads_executor_(std::move(threads_executor)) {}

  constexpr bool operator==(cublas_executor const& rhs) const noexcept {
    return device_ == rhs.device_ && handle_ == rhs.handle_ &&
           threads_executor_ == rhs.threads_executor_;
  }

  constexpr bool operator!=(cublas_executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  constexpr cublas_executor const& context() const noexcept {
    return *this;
  }

  // Implement the TwoWayExecutor interface.
  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F&& f, Ts&&... ts) const {
    return hpx::async(threads_executor_, internal::run_cublas, device_, handle_, std::forward<F>(f),
                      std::forward<Ts>(ts)...);
  }

private:
  int device_;
  cublasHandle_t handle_;
  hpx::threads::executors::pool_executor threads_executor_;
};

}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::cublas_executor> : std::true_type {};

}
}
}
