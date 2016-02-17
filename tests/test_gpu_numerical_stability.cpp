#define BOOST_TEST_MODULE GPU_NUMERICAL_STABILITY
#include "boost/test/unit_test.hpp"

#include <numeric>
#include <vector>

#include "test_utils.hpp"
#include "traits.hpp"
#include "padd_utils.h"
#include "image_stack_utils.h"
#include "convolution3Dfft.h"


namespace fc = fourierconvolution;

struct huge_fixture {

  fc::image_stack stack;
  fc::image_stack kernel;

  template <typename T>
  huge_fixture(const std::vector<T>& _stack_shape,
	       const std::vector<T>& _kernel_shape):
    stack(_stack_shape),
    kernel(_kernel_shape){

    std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
    std::fill(stack.data(),stack.data()+stack.num_elements(),42);
  }
  
};


BOOST_AUTO_TEST_SUITE(ramp)


BOOST_AUTO_TEST_CASE(times_two_128) {

  //define input data
  std::vector<size_t> k_shape(3,31);
  k_shape[fc::row_major::z] = 91;
  std::vector<size_t> s_shape(3,128);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 2;
  std::fill(stack.data(),stack.data()+stack.num_elements(),42);


  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
                              kernel.data(), &int_k_shape[0],
                              selectDeviceWithHighestComputeCapability());

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  for(size_t i = 0;i<expected.num_elements();++i)
    expected.data()[i] *= 2;

  double l2norm = std::inner_product(convolved.data(),
				     convolved.data() + convolved.num_elements(),
				     expected.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );
  l2norm /= stack.num_elements();
  const double l2_threshold = 1e-4;
  const bool result = l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< l2norm <<" not smaller than " << l2_threshold);
  BOOST_TEST_MESSAGE("[" << boost::unit_test::framework::current_test_case << "]\texpected\n" << fc::stack_to_string(expected)  << "\nreceived\n\n" << fc::stack_to_string(convolved) << "\n");
}

BOOST_AUTO_TEST_CASE(ramp_with_tiny_kernel_times_two) {

  //define input data
  std::vector<size_t> k_shape(3,3);
  k_shape[fc::row_major::z] = 5;
  std::vector<size_t> s_shape(3,128);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 2;

  for( size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tpadding input to    shape(x,y,z)=" << int_s_shape[fc::row_major::x]<< ", " << int_s_shape[fc::row_major::y]<< ", " << int_s_shape[fc::row_major::z]);
  
  //do convolution
  convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
                              kernel.data(), &int_k_shape[0],
                              selectDeviceWithHighestComputeCapability());

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  fc::image_stack expected(stack);
  for(size_t i = 0;i<expected.num_elements();++i)
    expected.data()[i] = 2*i;

  double l2norm = std::inner_product(convolved.data(),
				     convolved.data() + convolved.num_elements(),
				     expected.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );
  l2norm /= stack.num_elements();
  double l2_threshold = 1e-4;
  if(cuda_version() == 7050)
    l2_threshold = 1e-4;
  
  const bool result = l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< l2norm <<" not smaller than " << l2_threshold);
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tdump convolved by identity kernel\n\n" << fc::stack_to_string(convolved) << "\n");
  BOOST_TEST_MESSAGE("[" << boost::unit_test::framework::current_test_case << "]\texpected\n" << fc::stack_to_string(expected)  << "\nreceived\n\n" << fc::stack_to_string(convolved) << "\n");
}



BOOST_AUTO_TEST_CASE(ramp_with_tiny_kernel_times_two_padd_by_10fold) {

  //define input data
  std::vector<size_t> k_shape(3,3);
  k_shape[fc::row_major::z] = 5;
  std::vector<size_t> s_shape(3,128);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 2;

  for( size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape(),10);
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tpadding input to    shape(x,y,z)=" << int_s_shape[fc::row_major::x]<< ", " << int_s_shape[fc::row_major::y]<< ", " << int_s_shape[fc::row_major::z]);
  
  //do convolution
  convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
                              kernel.data(), &int_k_shape[0],
                              selectDeviceWithHighestComputeCapability());

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  fc::image_stack expected(stack);
  for(size_t i = 0;i<expected.num_elements();++i)
    expected.data()[i] = kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2]*i;

  double l2norm = std::inner_product(convolved.data(),
				     convolved.data() + convolved.num_elements(),
				     expected.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );
  l2norm /= stack.num_elements();
  double l2_threshold = 1e-4;
  if(cuda_version() == 7050)
    l2_threshold = 1e-4;
  
  const bool result = l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< l2norm <<" not smaller than " << l2_threshold);
  BOOST_TEST_MESSAGE("[" << boost::unit_test::framework::current_test_case << "]\texpected\n" << fc::stack_to_string(expected)  << "\nreceived\n\n" << fc::stack_to_string(convolved) << "\n");
}


BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(ramp)

BOOST_AUTO_TEST_CASE(ramp_normalized_16) {

  //define input data
  std::vector<size_t> k_shape(3,31);
  k_shape[fc::row_major::z] = 91;
  std::vector<size_t> s_shape(3,16);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 1.f;

  const float scale = 1.f/stack.num_elements();
  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i*scale;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
                              kernel.data(), &int_k_shape[0],
                              selectDeviceWithHighestComputeCapability());

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  for(size_t i = 0;i<expected.num_elements();++i)
    expected.data()[i] *= 2;

  double l2norm = std::inner_product(convolved.data(),
				     convolved.data() + convolved.num_elements(),
				     expected.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );
  l2norm /= stack.num_elements();
  const double l2_threshold = 1e-4;
  const bool result = l2norm<l2_threshold;
  
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< l2norm <<" not smaller than " << l2_threshold);
  BOOST_TEST_MESSAGE("[" << boost::unit_test::framework::current_test_case << "]\texpected\n" << fc::stack_to_string(expected)  << "\nreceived\n\n" << fc::stack_to_string(convolved) << "\n");
}

BOOST_AUTO_TEST_CASE(ramp_normalized_128) {

  //define input data
  std::vector<size_t> k_shape(3,31);
  k_shape[fc::row_major::z] = 91;
  std::vector<size_t> s_shape(3,128);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 1.f;

  const float scale = 1.f/stack.num_elements();
  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i*scale;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
                              kernel.data(), &int_k_shape[0],
                              selectDeviceWithHighestComputeCapability());

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  for(size_t i = 0;i<expected.num_elements();++i)
    expected.data()[i] *= 2;

  double l2norm = std::inner_product(convolved.data(),
				     convolved.data() + convolved.num_elements(),
				     expected.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );
  l2norm /= stack.num_elements();
  const double l2_threshold = 1e-4;
  
  const bool result = l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< l2norm <<" not smaller than " << l2_threshold);
}

BOOST_AUTO_TEST_CASE(ramp_normalized_512) {

  //define input data
  std::vector<size_t> k_shape(3,31);
  k_shape[fc::row_major::z] = 91;
  std::vector<size_t> s_shape(3,128);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 1.f;

  const float scale = 1.f/stack.num_elements();
  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i*scale;

  //create padded data and insert 
  fc::zero_padd<fc::image_stack> padder(stack.shape(), kernel.shape());
  fc::image_stack padded_stack(padder.extents_, stack.storage_order());
  padder.insert_at_offsets(stack, padded_stack);

  std::vector<int> int_s_shape(padder.extents_.rbegin(), padder.extents_.rend());
  std::vector<int> int_k_shape(k_shape.rbegin(), k_shape.rend());

  //do convolution
  convolution3DfftCUDAInPlace(padded_stack.data(), &int_s_shape[0],
                              kernel.data(), &int_k_shape[0],
                              selectDeviceWithHighestComputeCapability());

  //extract stack from padded_stack
  fc::image_stack convolved =   padded_stack
    [boost::indices
     [fc::range(padder.offsets()[fc::row_major::z], padder.offsets()[fc::row_major::z] + s_shape[fc::row_major::z])]
     [fc::range(padder.offsets()[fc::row_major::y], padder.offsets()[fc::row_major::y] + s_shape[fc::row_major::y])]
     [fc::range(padder.offsets()[fc::row_major::x], padder.offsets()[fc::row_major::x] + s_shape[fc::row_major::x])]];

  for(size_t i = 0;i<s_shape.size();++i)
    BOOST_REQUIRE_EQUAL(convolved.shape()[i],s_shape[i]);

  fc::image_stack expected(stack);
  for(size_t i = 0;i<expected.num_elements();++i)
    expected.data()[i] *= 2;

  double l2norm = std::inner_product(convolved.data(),
				     convolved.data() + convolved.num_elements(),
				     expected.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );
  l2norm /= stack.num_elements();
  const double l2_threshold = 1e-4;
  const bool result = l2norm<l2_threshold;
  BOOST_TEST_MESSAGE(boost::unit_test::framework::current_test_case << "\tconvolution3DfftCUDAInPlace    shape(x,y,z)=" << s_shape[fc::row_major::x]<< ", " << s_shape[fc::row_major::y]<< ", " << s_shape[fc::row_major::z] << "\tl2norm = " << l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< l2norm <<" not smaller than " << l2_threshold);
  BOOST_TEST_MESSAGE("[" << boost::unit_test::framework::current_test_case << "]\texpected\n" << fc::stack_to_string(expected)  << "\nreceived\n\n" << fc::stack_to_string(convolved) << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
