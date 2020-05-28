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

#include <hpx/include/async.hpp>
#include <hpx/include/dataflow.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/util/unwrap.hpp>
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
template <typename F, typename... Ts>
void run_cublas(int device, cublasHandle_t handle, F f, Ts... ts) noexcept {
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

/// An executor for CUBLAS functions.
///
/// Design rationale:
///
/// There are a few alternative HPX/CUDA implementation options:
///
/// 1) The approach taken here inserts an event just after a CUBLAS function is called and uses it to
///    track when the function has finished. Each task calls `hpx::util::yield_while()` and
///    `cudaEventQuery()` to infrom HPX's scheduler when it's is ready.
///
/// 2) The current implementation in HPX inserts a callback function on a stream using
///    `cudaStreamAddCallback()`. The callback registers the OS thread on which the stream is running
///    with HPX and sets the data associated with a custom future type to indicate that a preceding CUDA
///    function has completed.
///
///   There are a few issues with this approach:
///
///    - `cudaStreamAddCallback()` is deprecated in recent CUDA versions and is now superseded by
///      `cudaLaunchHostFunc()`.
///    - HPX throws an error if an external thread is registered multiple times, which happens as a
///      result of the callback being called multiple times. To make this work, the error is currently
///      ignored, which seems a little hacky.
///    - The design is based on `targets` instead of `executors`.
///
/// 3) [NOT IMPLEMENTED] This approach is similar to how MPI Futures are implemented. `cudaEvent_t` are
///    stored in a global array and polled for completion whenever a task yields or completes on a pool
///    that has the polling function registered.
///
/// It is unclear at the moment which approach is the best. According to John 1) and 3) ought to have
/// very similar performance. The two approaches are also available to MPI, performance comparisons
/// there did not produce a clear winner which appears to confirm John's statement.
///
/// Most CUDA functions are thread safe, that includes Streams and CUBLAS handles. The executor accepts a
/// `pool_executor` which specifies the host threads on which the CUBLAS calls are scheduled. There is a
/// price to pay for potentially sharing the CUBLAS handles between multiple threads:
///
///  - `cudaSetDevice()` needs to precede the CUBLAS call to make it's handle valid on that thread
///  - a mutex is needed to ensure the event is scheduled after the call and there is nothing in-between
///
/// We can eliminate both of these if we can dedicate a host thread to a particular device and only
/// execute CUBLAS / CUDA calls there. There are two approaches to achieve that with HPX:
///
/// a) separate pools with a single host threads per device
/// b) force bind tasks to OS threads
///
/// Apporach b) can work If either the static scheduler or John's scheduler (shared-priority-scheduler)
/// is used. Otherwise a hpx task is not guaranteed to actually run on the OS thread where it was
/// scheduled (it can get stolen). That is the case even if a schedule hint with the desired thread
/// number is provided. Note also that according to John, binding tasks within his scheduler is still
/// rather experimental.
///
/// Approach a) is the way to go but there are a few caveats. The following about HPX is good to know:
///
/// - Two HPX thread pools can't share OS threads
/// - The number OS threads used by HPX is fixed after it's initialized
/// - By default HPX doesn't allow oversubscribing cores with OS threads
///
/// Since a separate CUDA/CUBLAS host thread for a device will likely do little compared to other HPX
/// threads, we wouldn't want to dedicate a core to it. To overcome that, we have to enable thread
/// oversubscription in HPX and allow to schedule more than one HPX OS thread per core. We can do that via
/// the `hpx::resource::mode_allow_oversubscription` at initialization within the resource partitioner.
struct cublas_executor {
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  // The pool of host threads on which the calls to handle and device will be made.
  cublas_executor(cublas_pool& pool, bool pointer_mode_host)
      : device_(pool.device_id()), handle_(pool.handle()),
        threads_executor_("default", hpx::threads::thread_priority_high) {
    cublasPointerMode_t pointer_mode =
        (pointer_mode_host) ? CUBLAS_POINTER_MODE_HOST : CUBLAS_POINTER_MODE_DEVICE;
    DLAF_CUBLAS_CALL(cublasSetPointerMode(handle_, pointer_mode));
  }

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
  //
  // Note: the member can't be marked `const` because of `threads_executor_`.
  // Note: Parameters are passed by value as they are small types: pointers, integers or scalars.
  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F f, Ts... ts) {
    auto sched_f = [dev = this->device_, hdl = this->handle_, f, ts...] {
      internal::run_cublas<F, Ts...>(dev, hdl, f, ts...);
    };
    return hpx::async(threads_executor_, std::move(sched_f));
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
