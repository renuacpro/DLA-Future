#include <cublas_v2.h>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_init.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dlaf/cublas/executor.h"
#include "dlaf/cublas/pool.h"
#include "dlaf/cuda/pool.h"

// An executor defined on a single GPU.
//
int hpx_main(boost::program_options::variables_map&) {
  constexpr int device = 0;
  constexpr int num_streams = 2;
  dlaf::cuda::pool cuda_pool(device, num_streams);
  dlaf::cublas::pool cublas_pool(cuda_pool);

  dlaf::cublas::executor exec(cublas_pool, CUBLAS_POINTER_MODE_HOST);

  constexpr int n = 10000;
  constexpr int incx = 0;
  constexpr int incy = 0;
  constexpr double alpha = 2.0;

  // Initialize buffers on the device
  thrust::device_vector<double> x = thrust::host_vector<double>(n, 4.0);
  thrust::device_vector<double> y = thrust::host_vector<double>(n, 2.0);

  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy
  hpx::future<void> fut =
      exec.async_execute(cublasDaxpy, n, &alpha, x.data().get(), incx, y.data().get(), incy);

  thrust::host_vector<double> y_h = y;

  double sum_of_elems = 0;
  for (double e : y_h)
    sum_of_elems += e;

  std::cout << "result : " << sum_of_elems << std::endl;

  return hpx::finalize();
}

int main(int argc, char** argv) {
  return hpx::init(hpx_main, argc, argv);
}
