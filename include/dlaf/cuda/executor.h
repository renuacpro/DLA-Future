#pragma once

#include <c++/9.3.0/x86_64-pc-linux-gnu/bits/c++config.h>
#include <cstddef>
#include <utility>
#include <vector>

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

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
// The executor calls `setCudaDevice()` multiple times on each thread, any previous assignment of CUDA
// devices to threads is not preserved.
//
// Note: It is not recommended that multiple thread share the same CUBLAS handle because extreme care
//       needs to be taken when changing or destroying the handle.
template <typename F, typename... Ts>
void call_cublas(int device, cublasHandle_t handle, F&& f, Ts&&... ts) noexcept {
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

// For static frameworks the recommended best practice is to create all the needed streams upfront and
// destroy them after the work is done. This pattern is not immediately applicable to PyTorch, but a per
// device stream pool would achieve a similar effect. In particular, it would:
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
/// TODO: clarify Streams are a FIFO structure [1].
//
// [1]: CUDA Programming Guide, 3.2.5.5.4. Implicit Synchronization
//
// For multi-threaded applications that use the same device from different threads, the recommended
// programming model is to create one CUBLAS handle per thread and use that CUBLAS handle for the entire
// life of the thread. [1]
//
// [1]: cuBLAS Guide, 2.4 cuBLAS Helper Function Reference, cublasCreate()
//
class cublas_pool {
  // The number of streams spawned per device depends on the supported maximum number of concurrent
  // kernels that can execute in parallel on that device [1]. The number varies between different GPU
  // models, most support at least 16. The current number is chosen to be more conservative but can be
  // easily revised.
  //
  // [1]: CUDA Programming Guide, Table 15. Technical Specifications per Compute Capability
  static constexpr std::size_t num_streams_per_device = 5;
  std::vector<cublasHandle_t> handles_arr;

public:
  // This constructor can be used to specify the devices to which CUBLAS handles and streams should be
  // initialized. The constructor is useful if only a subset of the devices are used for CUBLAS calls.
  cublas_pool(std::vector<int> const& devices) noexcept {
    int num_devices = numDevices();
    handles.reserve(num_streams_per_device * devices.size());

    for (device : devices) {
      for (int i_stream = 0; i_stream < num_streams_per_device; ++i_stream) {
        cudaSetDevice(device);
        // - A kernel launch will fail if it is issued to a stream that is not associated to the current
        //   device. [1]
        //
        // [1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        handles.push_back(handle);
      }
    }
  }

  inline void printDeviceInfo(int device) const noexcept {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
  }

  inline int numDevices() const noexcept {
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    return num_devices;
  }

  ~cublas_pool() noexcept {
    for (cublasHandle_t handle : handles_arr) {
      // This implicitly calls `cublasDeviceSynchronize()` [1].
      //
      // [1]: cuBLAS, section 2.4 cuBLAS Helper Function Reference
      cublasDestroy(handle);
    }
  }
};

// An executor for CUBLAS functions.
struct cublas_executor {
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  // The pool of host threads on which the calls to handle and device will be made.
  cublas_executor(const std::string& pool_name, hpx::threads::thread_priority priority, int device,
                  cublasHandle_t handle)
      : pool_exec_(pool_name, priority), device_(device), handle_(handle) {}

  constexpr bool operator==(cublas_executor const& rhs) const noexcept {
    return device_ == rhs.device_ && handle_ == rhs.handle_ && pool_exec_ == rhs.pool_exec_;
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
    return hpx::async(pool_exec_, internal::cublas_call, device_, handle_, std::forward<F>(f),
                      std::forward<Ts>(ts)...);
  }

private:
  hpx::threads::executors::pool_executor pool_exec_;
  int device_;
  cublasHandle_t handle_;
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
