#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "convolution3Dfft.h"

BOOST_FIXTURE_TEST_SUITE(legacy_convolution,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(trivial_convolve) {

  float* image = image_.data();
  float* kernel = new float[kernel_size_];
  std::fill(kernel, kernel + kernel_size_, 0.f);

  convolution3DfftCUDAInPlace(image, &image_dims_[0], kernel, &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  float sum = std::accumulate(image, image + image_size_, 0.f);
  BOOST_CHECK_CLOSE(sum, 0.f, .00001);

  delete[] kernel;
}
BOOST_AUTO_TEST_SUITE_END()
