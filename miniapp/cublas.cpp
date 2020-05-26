#include <cublas_v2.h>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_init.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dlaf/cuda/executor.h"
#include "dlaf/cuda/pool.h"

int hpx_main(boost::program_options::variables_map&) {
  constexpr int device = 0;
  constexpr int num_streams = 2;
  dlaf::cublas_pool pool(device, num_streams);

  hpx::threads::executors::pool_executor threads_exec("default");
  dlaf::cublas_executor exec(pool, CUBLAS_POINTER_MODE_HOST, std::move(threads_exec));

  constexpr int n = 10000;
  constexpr int incx = 0;
  constexpr int incy = 0;

  // Initialize buffers on the device
  thrust::device_vector<double> x = thrust::host_vector<double>(n, 4.0);
  thrust::device_vector<double> y = thrust::host_vector<double>(n, 2.0);

  double result;
  hpx::future<void> fut =
      exec.async_execute(cublasDdot, n, x.data().get(), incx, y.data().get(), incy, &result);

  fut.get();
  std::cout << "result : " << result << std::endl;

  return hpx::finalize();
}

int main(int argc, char** argv) {
  return hpx::init(hpx_main, argc, argv);
}
