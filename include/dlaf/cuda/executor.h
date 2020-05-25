#pragma once

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
// https://github.com/rapidsai/rmm/issues/352
// TODO: Note: Streams are thread safe.
//
// https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDAStream.h
// Create a pool for cublas_handles associated to different devices and streams.
//
// TODO: spawn a separate thread for each device.
//
// TODO: clarify implicit synchronizations ?
// TODO: clarify Streams are a FIFO structure [1].
//
// [1]: CUDA Programming Guide, 3.2.5.5.4. Implicit Synchronization
//
// The pool is not thread safe!
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

  // The number of streams spawned per device depends on the supported maximum number of concurrent
  // kernels that can execute in parallel on that device [1]. The number varies between different GPU
  // models, most support at least 16.
  //
  // [1]: CUDA Programming Guide, Table 15. Technical Specifications per Compute Capability
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

  /// Return handles in Round-Robin.
  cublasHandle_t handle() noexcept {
    cublasHandle_t hd = handles_arr_[static_cast<std::size_t>(curr_handle_id_)];
    ++curr_handle_id_;
    if (curr_handle_id_ == num_handles())
      curr_handle_id_ = 0;
    return hd;
  }

  /// Return the current handle ID
  int current_handle_id() const noexcept {
    return curr_handle_id_;
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

  int num_handles() const noexcept {
    return static_cast<int>(handles_arr_.size());
  }
};

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
