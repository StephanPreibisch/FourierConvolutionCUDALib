#define BOOST_TEST_MODULE GPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include "padd_utils.h"
#include <numeric>

#include "convolution3Dfft.h"



BOOST_FIXTURE_TEST_SUITE(legacy_convolution,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(trivial_convolve) {

  float* image = image_.data();
  std::vector<float> kernel(kernel_size_,0);

  convolution3DfftCUDAInPlace(image, &image_dims_[0], 
			      &kernel[0], &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  float sum = std::accumulate(image, image + image_size_, 0.f);
  BOOST_CHECK_CLOSE(sum, 0.f, .00001);

}


BOOST_AUTO_TEST_CASE(identity_convolve) {

  using namespace multiviewnative;

  float sum_expected = std::accumulate(
      image_.data(), image_.data() + image_.num_elements(), 0.f);

  zero_padd<image_stack> padder(image_.shape(), identity_kernel_.shape());
  image_stack padded_image(padder.extents_, image_.storage_order());
  padder.insert_at_offsets(image_, padded_image);

  std::vector<int> extents_as_int(padder.extents_.size());
  std::copy(padder.extents_.begin(), padder.extents_.end(),
            extents_as_int.begin());

  convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0],
                              identity_kernel_.data(), &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);
  BOOST_CHECK_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_CASE(horizontal_convolve) {
  using namespace multiviewnative;

  float sum_expected =
      std::accumulate(image_folded_by_horizontal_.data(),
                      image_folded_by_horizontal_.data() +
                          image_folded_by_horizontal_.num_elements(),
                      0.f);

  zero_padd<image_stack> padder(image_.shape(), horizont_kernel_.shape());
  image_stack padded_image(padder.extents_, image_.storage_order());

  padder.insert_at_offsets(image_, padded_image);

  std::vector<int> extents_as_int(padder.extents_.size());
  std::copy(padder.extents_.begin(), padder.extents_.end(),
            extents_as_int.begin());

  convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0],
                              horizont_kernel_.data(), &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  image_ = padded_image
      [boost::indices
           [range(padder.offsets()[0], padder.offsets()[0] + image_dims_[0])]
           [range(padder.offsets()[1], padder.offsets()[1] + image_dims_[1])]
           [range(padder.offsets()[2], padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);

  BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_CASE(vertical_convolve) {

  multiviewnative::zero_padd<multiviewnative::image_stack> padder(
      image_.shape(), vertical_kernel_.shape());
  multiviewnative::image_stack padded_image(padder.extents_,
                                            image_.storage_order());

  padder.insert_at_offsets(image_, padded_image);

  std::vector<int> extents_as_int(padder.extents_.size());
  std::copy(padder.extents_.begin(), padder.extents_.end(),
            extents_as_int.begin());

  convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0],
                              vertical_kernel_.data(), &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  float sum_expected =
      std::accumulate(image_folded_by_vertical_.data(),
                      image_folded_by_vertical_.data() +
                          image_folded_by_vertical_.num_elements(),
                      0.f);

  image_ = padded_image
      [boost::indices
           [multiviewnative::range(padder.offsets()[0],
                                   padder.offsets()[0] + image_dims_[0])]
           [multiviewnative::range(padder.offsets()[1],
                                   padder.offsets()[1] + image_dims_[1])]
           [multiviewnative::range(padder.offsets()[2],
                                   padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);
  BOOST_CHECK_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_CASE(depth_convolve) {

  multiviewnative::zero_padd<multiviewnative::image_stack> padder(
      image_.shape(), depth_kernel_.shape());
  multiviewnative::image_stack padded_image(padder.extents_,
                                            image_.storage_order());

  padder.insert_at_offsets(image_, padded_image);

  std::vector<int> extents_as_int(padder.extents_.size());
  std::copy(padder.extents_.begin(), padder.extents_.end(),
            extents_as_int.begin());

  convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0],
                              depth_kernel_.data(), &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  float sum_expected = std::accumulate(
      image_folded_by_depth_.data(),
      image_folded_by_depth_.data() + image_folded_by_depth_.num_elements(),
      0.f);

  image_ = padded_image
      [boost::indices
           [multiviewnative::range(padder.offsets()[0],
                                   padder.offsets()[0] + image_dims_[0])]
           [multiviewnative::range(padder.offsets()[1],
                                   padder.offsets()[1] + image_dims_[1])]
           [multiviewnative::range(padder.offsets()[2],
                                   padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);
  BOOST_CHECK_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_CASE(all1_convolve) {

  multiviewnative::zero_padd<multiviewnative::image_stack> padder(
      image_.shape(), all1_kernel_.shape());
  multiviewnative::image_stack padded_image(padder.extents_,
                                            image_.storage_order());

  padder.insert_at_offsets(image_, padded_image);

  std::vector<int> extents_as_int(padder.extents_.size());
  std::copy(padder.extents_.begin(), padder.extents_.end(),
            extents_as_int.begin());

  convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0],
                              all1_kernel_.data(), &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  float sum_expected = std::accumulate(
      image_folded_by_all1_.data(),
      image_folded_by_all1_.data() + image_folded_by_all1_.num_elements(), 0.f);

  image_ = padded_image
      [boost::indices
           [multiviewnative::range(padder.offsets()[0],
                                   padder.offsets()[0] + image_dims_[0])]
           [multiviewnative::range(padder.offsets()[1],
                                   padder.offsets()[1] + image_dims_[1])]
           [multiviewnative::range(padder.offsets()[2],
                                   padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);
  BOOST_CHECK_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_SUITE_END()
