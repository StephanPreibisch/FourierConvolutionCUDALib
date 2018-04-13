
#include "benchmark/benchmark.h"

#include "test_fixtures.hpp"
#include "padd_utils.h"

#include "convolution3Dfft.h"
#include "test_utils.hpp"
#include "image_stack_utils.h"
#include "traits.hpp"

#include <vector>

namespace fc = fourierconvolution;


static void BM_simple_fp32(benchmark::State& state) {

  fc::default_3D_fixture fix;

  std::vector<int> image_dims(3,64);
  std::size_t image_len = std::pow(64,3);
  std::vector<float> image(image_len,0.);

  std::vector<int> kernel_dims(3,3);
  std::size_t kernel_len = std::pow(3,3);
  std::vector<float> kernel(kernel_len,0);

  while (state.KeepRunning()){

    convolution3DfftCUDAInPlace(&image[0], &image_dims[0] ,
                                &kernel[0], &kernel_dims[0] ,
                                selectDeviceWithHighestComputeCapability());
  }

}

BENCHMARK(BM_simple_fp32);
BENCHMARK_MAIN();

// BOOST_FIXTURE_TEST_SUITE(legacy_convolution,
//                          fc::default_3D_fixture)

// BOOST_AUTO_TEST_CASE(trivial_convolve) {

//   float* image = image_.data();
//   std::vector<float> kernel(kernel_size_,0);
  
//   convolution3DfftCUDAInPlace(image, &image_dims_[0],
// 			      &kernel[0], &kernel_dims_[0],
//                               selectDeviceWithHighestComputeCapability());

//   float sum = std::accumulate(image, image + image_size_, 0.f);
//   BOOST_CHECK_CLOSE(sum, 0.f, .00001);

// }
