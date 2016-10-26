#define BOOST_TEST_MODULE GPU_CONVOLUTION
#include <numeric>

#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include "padd_utils.h"

#include "convolution3Dfft.h"
#include "test_utils.hpp"
#include "image_stack_utils.h"
#include "traits.hpp"

namespace fc = fourierconvolution;

BOOST_FIXTURE_TEST_SUITE(legacy_convolution,
                         fc::default_3D_fixture)

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

  // using namespace fourierconvolution;

  float sum_expected = std::accumulate(
      image_.data(), image_.data() + image_.num_elements(), 0.f);

  fc::zero_padd<fc::image_stack> padder(image_.shape(), identity_kernel_.shape());
  fc::image_stack padded_image(padder.extents_, image_.storage_order());
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
  // using namespace fourierconvolution;

  float sum_expected =
      std::accumulate(image_folded_by_horizontal_.data(),
                      image_folded_by_horizontal_.data() +
                          image_folded_by_horizontal_.num_elements(),
                      0.f);

  fc::zero_padd<fc::image_stack> padder(image_.shape(), horizont_kernel_.shape());
  fc::image_stack padded_image(padder.extents_, image_.storage_order());

  padder.insert_at_offsets(image_, padded_image);

  std::vector<int> extents_as_int(padder.extents_.size());
  std::copy(padder.extents_.begin(), padder.extents_.end(),
            extents_as_int.begin());

  convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0],
                              horizont_kernel_.data(), &kernel_dims_[0],
                              selectDeviceWithHighestComputeCapability());

  image_ = padded_image
      [boost::indices
           [fc::range(padder.offsets()[0], padder.offsets()[0] + image_dims_[0])]
           [fc::range(padder.offsets()[1], padder.offsets()[1] + image_dims_[1])]
           [fc::range(padder.offsets()[2], padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);

  BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_CASE(vertical_convolve) {

  fc::zero_padd<fc::image_stack> padder(
      image_.shape(), vertical_kernel_.shape());
  fc::image_stack padded_image(padder.extents_,
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
           [fc::range(padder.offsets()[0],
                                   padder.offsets()[0] + image_dims_[0])]
           [fc::range(padder.offsets()[1],
                                   padder.offsets()[1] + image_dims_[1])]
           [fc::range(padder.offsets()[2],
                                   padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);
  BOOST_CHECK_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_CASE(depth_convolve) {

  fc::zero_padd<fc::image_stack> padder(
      image_.shape(), depth_kernel_.shape());
  fc::image_stack padded_image(padder.extents_,
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
           [fc::range(padder.offsets()[0],
                                   padder.offsets()[0] + image_dims_[0])]
           [fc::range(padder.offsets()[1],
                                   padder.offsets()[1] + image_dims_[1])]
           [fc::range(padder.offsets()[2],
                                   padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);
  BOOST_CHECK_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_CASE(all1_convolve) {

  fc::zero_padd<fc::image_stack> padder(
      image_.shape(), all1_kernel_.shape());
  fc::image_stack padded_image(padder.extents_,
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
           [fc::range(padder.offsets()[0],
                                   padder.offsets()[0] + image_dims_[0])]
           [fc::range(padder.offsets()[1],
                                   padder.offsets()[1] + image_dims_[1])]
           [fc::range(padder.offsets()[2],
                                   padder.offsets()[2] + image_dims_[2])]];

  float sum = std::accumulate(image_.data(),
                              image_.data() + image_.num_elements(), 0.f);
  BOOST_CHECK_CLOSE(sum, sum_expected, .00001);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(asymmetric_volumes)

BOOST_AUTO_TEST_CASE(identity_convolve_of_prime_shape) {

  //define input data
  std::vector<size_t> k_shape(3,3);

  std::vector<size_t> s_shape(3,16);
  s_shape[fc::row_major::x] = 13;
  s_shape[fc::row_major::y] = 17;
  s_shape[fc::row_major::z] = 19;
  
  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);

  size_t where_z = k_shape[fc::row_major::z]/2;
  size_t where_y = k_shape[fc::row_major::y]/2;
  size_t where_x = k_shape[fc::row_major::x]/2;
  
  kernel[where_z][where_y][where_x] = 1;
  
  for( size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  try{
    convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
				kernel.data(), &int_k_shape[0],
				selectDeviceWithHighestComputeCapability()
				);
  }
  catch(std::runtime_error& exc){
    BOOST_FAIL("failed due to " << exc.what()
	       << "\n do not use convolution3DfftCUDAInPlace on this GPU with shapes larger than "
	       << "(x,y,z) = " << s_shape[fc::row_major::x] << "x" << s_shape[fc::row_major::y] << "x" << s_shape[fc::row_major::z]);
  }

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  
  double my_l2norm = l2norm(convolved,expected);
  const double l2_threshold = 1e-4;
  const bool result = my_l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case().p_name << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << l2_threshold);

}

BOOST_AUTO_TEST_CASE(horizontal_convolve_of_prime_shape) {

  //define input data
  std::vector<size_t> k_shape(3,3);

  std::vector<size_t> s_shape(3,16);
  s_shape[fc::row_major::x] = 13;
  s_shape[fc::row_major::y] = 17;
  s_shape[fc::row_major::z] = 19;
  
  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  std::fill(stack.data(),stack.data()+stack.num_elements(),1);

  size_t where_z = k_shape[fc::row_major::z]/2;
  size_t where_y = k_shape[fc::row_major::y]/2;
  size_t where_x = k_shape[fc::row_major::x]/2;
  
  kernel[where_z][where_y][where_x] = 1;
  kernel[where_z][where_y][where_x-1] = 1;
  kernel[where_z][where_y][where_x+1] = 1;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  try{
    convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
				kernel.data(), &int_k_shape[0],
				selectDeviceWithHighestComputeCapability()
				);
  }
  catch(std::runtime_error& exc){
    BOOST_FAIL("failed due to " << exc.what()
	       << "\n do not use convolution3DfftCUDAInPlace on this GPU with shapes larger than "
	       << "(x,y,z) = " << s_shape[fc::row_major::x] << "x" << s_shape[fc::row_major::y] << "x" << s_shape[fc::row_major::z]);
  }

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  std::fill(expected.data(),expected.data()+expected.num_elements(),3);

  double my_l2norm = l2norm(convolved,expected);
  const double l2_threshold = 1e-2;
  const bool result = my_l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case().p_name << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << l2_threshold);

}

BOOST_AUTO_TEST_CASE(vertical_convolve_of_prime_shape) {

  //define input data
  std::vector<size_t> k_shape(3,3);

  std::vector<size_t> s_shape(3,16);
  s_shape[fc::row_major::x] = 13;
  s_shape[fc::row_major::y] = 17;
  s_shape[fc::row_major::z] = 19;
  
  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  std::fill(stack.data(),stack.data()+stack.num_elements(),1);

  size_t where_z = k_shape[fc::row_major::z]/2;
  size_t where_y = k_shape[fc::row_major::y]/2;
  size_t where_x = k_shape[fc::row_major::x]/2;
  
  kernel[where_z][where_y][where_x] = 1;
  kernel[where_z][where_y-1][where_x] = 1;
  kernel[where_z][where_y+1][where_x] = 1;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  try{
    convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
				kernel.data(), &int_k_shape[0],
				selectDeviceWithHighestComputeCapability()
				);
  }
  catch(std::runtime_error& exc){
    BOOST_FAIL("failed due to " << exc.what()
	       << "\n do not use convolution3DfftCUDAInPlace on this GPU with shapes larger than "
	       << "(x,y,z) = " << s_shape[fc::row_major::x] << "x" << s_shape[fc::row_major::y] << "x" << s_shape[fc::row_major::z]);
  }

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  std::fill(expected.data(),expected.data()+expected.num_elements(),3);
  
  double my_l2norm = l2norm(convolved,expected);
    
  const double l2_threshold = 2e-2;
  const bool result = my_l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case().p_name << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << l2_threshold);

}

BOOST_AUTO_TEST_CASE(diagonal_convolve_of_prime_shape) {

  //define input data
  size_t kernel_diameter = 3;
  std::vector<size_t> k_shape(3,kernel_diameter);

  std::vector<size_t> s_shape(3,16);
  s_shape[fc::row_major::x] = 13;
  s_shape[fc::row_major::y] = 17;
  s_shape[fc::row_major::z] = 19;
  
  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  std::fill(stack.data(),stack.data()+stack.num_elements(),1);

  for(int i = 0;i<kernel_diameter;++i)
    kernel[i][i][i] = 1;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  try{
    convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
				kernel.data(), &int_k_shape[0],
				selectDeviceWithHighestComputeCapability()
				);
  }
  catch(std::runtime_error& exc){
    BOOST_FAIL("failed due to " << exc.what()
	       << "\n do not use convolution3DfftCUDAInPlace on this GPU with shapes larger than "
	       << "(x,y,z) = " << s_shape[fc::row_major::x] << "x" << s_shape[fc::row_major::y] << "x" << s_shape[fc::row_major::z]);
  }

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  std::fill(expected.data(),expected.data()+expected.num_elements(),kernel_diameter*stack.data()[0]);
  
  double my_l2norm = l2norm(convolved,expected);
    
  const double l2_threshold = 2e-2;
  const bool result = my_l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case().p_name << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << l2_threshold);

}
BOOST_AUTO_TEST_SUITE_END()
