#define BOOST_TEST_MODULE GPU_MEMORY_REACH
#include "boost/test/unit_test.hpp"

//#include "padd_utils.h"
#include <numeric>
#include <vector>

#include "padd_utils.h"
#include "image_stack_utils.h"
#include "convolution3Dfft.h"

namespace fourierconvolution {

  struct row_major {

    const static size_t x = 2;
    const static size_t y = 1;
    const static size_t z = 0;

    const static size_t w = x;
    const static size_t h = y;
    const static size_t d = z;
  };

};

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


BOOST_AUTO_TEST_SUITE(memory_broken)

BOOST_AUTO_TEST_CASE(c_storage_order) {


  std::vector<size_t> k_shape(3,31);
  k_shape[fc::row_major::z] = 91;
  std::vector<size_t> k_halfshape(k_shape);
  for(int i = 0;i<k_halfshape.size();++i)
    k_halfshape[i] /= 2;
  
  fc::image_stack kernel(k_shape);

  
  BOOST_REQUIRE_EQUAL(kernel.shape()[fc::row_major::z],k_shape[fc::row_major::z]);
  BOOST_REQUIRE_EQUAL(kernel.shape()[fc::row_major::x],k_shape[fc::row_major::x]);

  size_t position = 0;
  for (;position<kernel.num_elements();++position)
    kernel.data()[position] = position;
  
  BOOST_CHECK_EQUAL(kernel[1][0][0],k_shape[fc::row_major::h]*k_shape[fc::row_major::w]);
  BOOST_CHECK_EQUAL(kernel[2][0][0],2*k_shape[fc::row_major::h]*k_shape[fc::row_major::w]);
  BOOST_CHECK_EQUAL(kernel[2][1][0],(2*k_shape[fc::row_major::h]*k_shape[fc::row_major::w])+k_shape[fc::row_major::w]);
  BOOST_CHECK_EQUAL(kernel[2][2][0],(2*k_shape[fc::row_major::h]*k_shape[fc::row_major::w])+2*k_shape[fc::row_major::w]);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_halfshape[fc::row_major::z]][k_halfshape[fc::row_major::y]][k_halfshape[fc::row_major::x]] = 1;

  position = 0;
  for (;position<kernel.num_elements();++position){
    if(kernel.data()[position])
      break;
  }

  size_t central_pixel	=  k_shape[fc::row_major::h]*k_shape[fc::row_major::w]*k_halfshape[fc::row_major::z];

  central_pixel 	+= k_shape[fc::row_major::w]*k_halfshape[fc::row_major::y];
  central_pixel 	+= k_halfshape[fc::row_major::x];
  
  BOOST_CHECK_EQUAL(central_pixel,position);
  BOOST_CHECK_EQUAL(kernel.data()[central_pixel],1);
  
  size_t central_expected = (31*31*(91/2)) + (31*(31/2)) + (31/2);
  BOOST_REQUIRE_EQUAL(central_pixel,central_expected);



}

BOOST_AUTO_TEST_CASE(identity_convolve_512) {

  //define input data
  std::vector<size_t> k_shape(3,31);
  k_shape[fc::row_major::z] = 91;
  std::vector<size_t> s_shape(3,512);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 1;
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

  BOOST_CHECK_EQUAL(padded_stack[int_s_shape[fc::row_major::d]/2][int_s_shape[fc::row_major::h]/2][int_s_shape[fc::row_major::w]/2],42);
}

BOOST_AUTO_TEST_CASE(times_two_512) {

  //define input data
  std::vector<size_t> k_shape(3,31);
  k_shape[fc::row_major::z] = 91;
  std::vector<size_t> s_shape(3,512);

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

  for(size_t i = 0;i<stack.num_elements();++i)
    BOOST_REQUIRE_MESSAGE(convolved.data()[i]==2*stack.data()[i], "convolved " << convolved.data()[i] << " not equal expected " << 2*stack.data()[i] << " at " << i );
}

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

  for(size_t i = 0;i<stack.num_elements();++i)
    BOOST_REQUIRE_MESSAGE(convolved.data()[i]==2*stack.data()[i], "convolved " << convolved.data()[i] << " not equal expected " << 2*stack.data()[i] << " at " << i );
}

BOOST_AUTO_TEST_CASE(ramp_with_tiny_kernel) {

  //define input data
  std::vector<size_t> k_shape(3,3);
  k_shape[fc::row_major::z] = 5;
  std::vector<size_t> s_shape(3,128);

  fc::image_stack kernel(k_shape);
  fc::image_stack stack(s_shape);

  std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
  kernel[k_shape[fc::row_major::z]/2][k_shape[fc::row_major::y]/2][k_shape[fc::row_major::x]/2] = 2;
  std::fill(stack.data(),stack.data()+stack.num_elements(),42);
  for( size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

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

  for(size_t i = 0;i<stack.num_elements();++i){
    double diff = convolved.data()[i]-i;
    BOOST_CHECK_MESSAGE(std::abs(diff)<1e-2, "convolved " << convolved.data()[i] << " not equal expected " << 2*stack.data()[i] << " at " << i );
  }
}

BOOST_AUTO_TEST_CASE(all_tiny) {

  //define input data
  std::vector<size_t> k_shape(3,3);
  k_shape[fc::row_major::z] = 5;
  std::vector<size_t> s_shape(3,16);

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

  for(size_t i = 0;i<stack.num_elements();++i){
    double diff = convolved.data()[i]-stack.data()[i];
    BOOST_REQUIRE_MESSAGE(std::abs(diff)<1e-2, "convolved " << convolved.data()[i] << " not equal expected " << 2*stack.data()[i] << " at " << i );
  }
}

BOOST_AUTO_TEST_SUITE_END()
