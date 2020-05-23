#pragma once

#include <utility>

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/util/yield_while.hpp>

#include "dlaf/cuda/error.h"

namespace dlaf {

// - A kernel launch will fail if it is issued to a stream that is not associated to the current
//   device. [1]
namespace internal {

// Call the CUBLAS function and schedule an event after it to query when it's done.
template <typename F, typename... Ts>
void call_cublas(int device, cublasHandle_t handle, F&& f, Ts&&... ts) noexcept {
  // Set the device corresponding to the CUBLAS handle.
  //
  // The CUBLAS library context is tied to the current CUDA device [2]. A previous task scheduled on the
  // same thread may have set a different device, this makes sure the correct device is used. The
  // function is considered very low overhead call [3].
  //
  // [1]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
  // [2]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
  // [3]: CUDA Runtime API, section 5.1 Device Management
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

// TODO: Do I need copy/move constructors?
// TODO: use all GPUs by default but allow to specify less gpus
// TODO: set pointer mode?
// TODO: A new cublas context needs to be created if the device is reset
// TODO: there needs to be a mutex between to ensure nothing is scheduled in-between
// TODO: Spawn a task here !!
// TODO: Make sure the stream is associated with the device
// TODO: set DLAF_CUBLAS_CALL(cublasSetStream(handle_, stream)); when the CUBLAS handle is created
// TODO: CUDA_LAUNCH_BLOCKING environment variable used for profiling?
// TODO: implicit synchronizations ?
// TODO: CUDA_API_PER_THREAD_DEFAULT_STREAM ??
// TODO: consider using `cudaStreamCreateWithFlags()` with `cudaStreamNonBlocking` to get concurrency
// from libraries to which you don't have full control.

// Note: Streams are thread safe.
//
// https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDAStream.h

// Maximum number of concurrent kernel execution :
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
// Most have at least 16

// TODO: It is advisable to minimize calls to `cublasCreate()` and `cublasDestroy()`. Note that
// `cublasDeviceSynchronize()` is called when `cublasDestroy()` is called
//
// For multi-threaded applications that use the same device from different threads, the recommended
// programming model is to create one CUBLAS handle per thread and use that CUBLAS handle for the entire
// life of the thread.

// Streams are a FIFO structure

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization

// ** On the cost of stream creation and destruction
//
// For static frameworks the recommended best practice is to create all the needed streams upfront and
// destroy them after the work is done. This pattern is not immediately applicable to PyTorch, but a per
// device stream pool would achieve a similar effect. In particular, it would:
//
// https://github.com/pytorch/pytorch/issues/9646
// https://github.com/rapidsai/rmm/issues/352

// Alternative : For code that is compiled using the --default-stream per-thread compilation flag (or
// that defines the CUDA_API_PER_THREAD_DEFAULT_STREAM macro before including CUDA headers (cuda.h and
// cuda_runtime.h)), the default stream is a regular stream and each host thread has its own default
// stream.

// According to
// https://stackoverflow.com/questions/3565793/is-there-a-maximum-number-of-streams-in-cuda
// the maximum number of concurrent kernels executing on the Fermi GPU's is 16.

// Register this thread with HPX, this should be done once for each external OS-thread intended to invoke
// HPX functionality. Calling this function more than once on the same thread will report an error.
//
// Unregister the thread from HPX, this should be done once in
// the end before the external thread exists.
// hpx::error_code ec(hpx::lightweight);  // ignore errors
// hpx::register_thread(this_->rt_, "cuda", ec);
// this_->set_data(hpx::util::unused);
// hpx::unregister_thread(this_->rt_);

// Enqueue a host function call in a stream. The function will be called after currently enqueued
// work and will block work added after it.
//
// Note: CudaStreamAddCallback() was deprecated in CUDA 10
//
// DLAF_CUDA_CALL(cudaLaunchHostFunc(stream, stream_callback, this));

// It is not recommended that multiple thread share the same CUBLAS handle
// because extreme care needs to be taken when changing or destroying the
// handle.

// The executor calls `setCudaDevice()` multiple times on each thread, any previous assignment of CUDA
// devices to threads is not preserved.
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
