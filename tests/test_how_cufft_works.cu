#define BOOST_TEST_MODULE HOW_CUFFT_WORKS
#include "boost/test/unit_test.hpp"

#ifndef FC_TRACE
#define FC_TRACE false
#endif

#include <numeric>
#include <vector>

#include "test_utils.hpp"
#include "image_stack_utils.h"
#include "traits.hpp"
#include "book.h"
#include "cufft.h"

namespace fourierconvolution {

  
  __global__ void scale(cufftComplex* _array, size_t _size, float _scale){

    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    cufftComplex el;
    if(tid<_size){
      el = _array[tid];
      _array[tid].x = el.x*_scale;
      _array[tid].y = el.y*_scale;
    }
      

  }

  void inplace_fft_ifft(image_stack& _stack){

    
    typedef boost::multi_array<cufftComplex,3> frequ_stack;
    
    const size_t img_size = _stack.num_elements();
    std::vector<size_t> shape(_stack.shape(),_stack.shape() + image_stack::dimensionality);
    
    BOOST_REQUIRE(img_size > 32);
  
    std::vector<size_t> shape_for_cufft(shape);
    shape_for_cufft[row_major::x] = (shape[row_major::x]/2) + 1;
    const size_t size_for_cufft = std::accumulate(shape_for_cufft.begin(), shape_for_cufft.end(),1,std::multiplies<size_t>());
  
    cufftComplex* d_stack = 0;
  
    HANDLE_ERROR( cudaMalloc( (void**)&(d_stack), size_for_cufft*sizeof(cufftComplex) ) );
    HANDLE_ERROR( cudaMemset( d_stack, 0, size_for_cufft*sizeof(cufftComplex) ));

    //transform input data to cufft/fftw
    frequ_stack cufft_compliant(shape_for_cufft);
    float* stack_begin = _stack.data();
    float* cufft_begin = reinterpret_cast<float*>(cufft_compliant.data());
    
    for(size_t z = 0;z<shape[row_major::in_z];++z)
      for(size_t y = 0;y<shape[row_major::in_y];++y){
	
	size_t cufft_line_offset = (z*shape_for_cufft[row_major::in_y]*shape_for_cufft[row_major::in_x])+ (y*shape_for_cufft[row_major::in_x]);
	cufft_begin = reinterpret_cast<float*>(&cufft_compliant.data()[cufft_line_offset]);
	
	size_t stack_line_offset = (z*shape[row_major::in_y]*shape[row_major::in_x])+ (y*shape[row_major::in_x]);
	stack_begin = &_stack.data()[stack_line_offset];
	
	std::copy(stack_begin,stack_begin + shape[row_major::in_x],cufft_begin);
	
      }
    
    HANDLE_ERROR( cudaMemcpy( d_stack, cufft_compliant.data(), size_for_cufft*sizeof(cufftComplex) , cudaMemcpyHostToDevice ) );

    //FORWARD
    cufftHandle fftPlanFwd;
    cufftPlan3d(&fftPlanFwd, shape[row_major::x], shape[row_major::y], shape[row_major::z], CUFFT_R2C);HANDLE_ERROR_KERNEL;
    if(CUDART_VERSION < 6050)
      cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_FFTW_PADDING);

    cufftExecR2C(fftPlanFwd, (cufftReal*)d_stack, (cufftComplex *)d_stack);HANDLE_ERROR_KERNEL;
    ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;

    //apply scale
    const float scale_ = 1.f/float(img_size);
    unsigned threads = 32;
    unsigned blocks = (size_for_cufft + threads -1) /threads;
    scale<<<blocks,threads>>>(d_stack,size_for_cufft,scale_);
  
    //BACKWARD
    cufftHandle fftPlanInv;
    cufftPlan3d(&fftPlanInv, shape[row_major::x], shape[row_major::y], shape[row_major::z], CUFFT_C2R);HANDLE_ERROR_KERNEL;
    if(CUDART_VERSION < 6050)
      cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_FFTW_PADDING);
    
    cufftExecC2R(fftPlanInv, (cufftComplex*)d_stack, (cufftReal *)d_stack);HANDLE_ERROR_KERNEL;
    ( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;

    cufftComplex zero;zero.x = 0;zero.y = 0;
    std::fill(cufft_compliant.data(),cufft_compliant.data()+cufft_compliant.num_elements(),zero);
    HANDLE_ERROR( cudaMemcpy( cufft_compliant.data(), d_stack , size_for_cufft*sizeof(cufftComplex) , cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaFree( d_stack));

    for(size_t z = 0;z<shape[row_major::in_z];++z)
      for(size_t y = 0;y<shape[row_major::in_y];++y){
	
	size_t cufft_line_offset = (z*shape_for_cufft[row_major::in_y]*shape_for_cufft[row_major::in_x])+ (y*shape_for_cufft[row_major::in_x]);
	cufft_begin = reinterpret_cast<float*>(&cufft_compliant.data()[cufft_line_offset]);
	
	size_t stack_line_offset = (z*shape[row_major::in_y]*shape[row_major::in_x])+ (y*shape[row_major::in_x]);
	stack_begin = &_stack.data()[stack_line_offset];
	
	std::copy(cufft_begin,cufft_begin + shape[row_major::in_x],stack_begin);
	
      }

    return;
  }
  
  void outofplace_fft_ifft(const image_stack& _input,  image_stack& _output){

    
    std::vector<size_t> shape(_input.shape(),_input.shape() + 3);
    const size_t stack_size = _input.num_elements();

    if(_output.num_elements()!=stack_size)
      _output.resize(shape);
    
    std::fill(_output.data(),_output.data()+stack_size,0);
    
    std::vector<size_t> shape_for_cufft(shape);
    shape_for_cufft[row_major::x] = (shape[row_major::x]/2) + 1;
    size_t size_for_cufft = std::accumulate(shape_for_cufft.begin(), shape_for_cufft.end(),1,std::multiplies<size_t>());
  
    cufftComplex* d_complex = 0;
    cufftReal* d_real = 0;
  
    HANDLE_ERROR( cudaMalloc( (void**)&(d_complex), size_for_cufft*sizeof(cufftComplex) ) );
    HANDLE_ERROR( cudaMemset( d_complex, 0, size_for_cufft*sizeof(cufftComplex) ));

    HANDLE_ERROR( cudaMalloc( (void**)&(d_real), stack_size*sizeof(cufftComplex) ) );
    HANDLE_ERROR( cudaMemcpy( d_real, _input.data(), stack_size*sizeof(float) , cudaMemcpyHostToDevice ) );

    //FORWARD
    cufftHandle fftPlanFwd;
    cufftPlan3d(&fftPlanFwd, shape[row_major::x], shape[row_major::y], shape[row_major::z], CUFFT_R2C);HANDLE_ERROR_KERNEL;
    if(CUDART_VERSION < 6050)
      cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_FFTW_PADDING);
    cufftExecR2C(fftPlanFwd, d_real, d_complex);HANDLE_ERROR_KERNEL;

    //apply scale
    const float scale_ = 1.f/float(stack_size);
    unsigned threads = 32;
    unsigned blocks = (size_for_cufft + threads -1) /threads;
    scale<<<blocks,threads>>>(d_complex,size_for_cufft,scale_);
  
    //BACKWARD
    cufftHandle fftPlanInv;
    cufftPlan3d(&fftPlanInv, shape[row_major::x], shape[row_major::y], shape[row_major::z], CUFFT_C2R);HANDLE_ERROR_KERNEL;
    if(CUDART_VERSION < 6050)
      cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_FFTW_PADDING);

    cufftExecC2R(fftPlanInv, d_complex, d_real);HANDLE_ERROR_KERNEL;
  
    std::fill(_output.data(),_output.data()+stack_size,0);
    HANDLE_ERROR( cudaMemcpy( _output.data(), d_real , stack_size*sizeof(float) , cudaMemcpyDeviceToHost ) );

    ( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
    ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;

    HANDLE_ERROR( cudaFree( d_real));
    HANDLE_ERROR( cudaFree( d_complex));
  
  }
  
};

namespace fc = fourierconvolution;



BOOST_AUTO_TEST_SUITE(inplace)

BOOST_AUTO_TEST_CASE(of_prime_shape) {

  std::vector<size_t> shape(3,17);
  shape[fc::row_major::z] = 13;
  shape[fc::row_major::x] = 19;
  
  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  fc::image_stack received(stack);

  fc::inplace_fft_ifft(received);
  
  double my_l2norm = l2norm(stack,received);
  const double expected = 1e-1;
  const bool result = my_l2norm<expected;

  if(!result && FC_TRACE){
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n";
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
  }
  
  BOOST_TEST_MESSAGE("inplace    shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);
  
  
}

BOOST_AUTO_TEST_CASE(power_of_2) {

  std::vector<size_t> shape(3,16);
  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  fc::image_stack received(stack);

  fc::inplace_fft_ifft(received);
  
double my_l2norm = l2norm(stack,received);
  const double expected = 1e-1;
  const bool result = my_l2norm<expected;

  if(!result && FC_TRACE){
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n";
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
  }
  
  BOOST_TEST_MESSAGE("inplace    shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);
  
  
}


BOOST_AUTO_TEST_CASE(power_of_3) {

  std::vector<size_t> shape(3,27);
  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  fc::image_stack received(stack);

  fc::inplace_fft_ifft(received);
  
double my_l2norm = l2norm(stack,received);
  const double expected = 1e-1;
  const bool result = my_l2norm<expected;

  if(!result && FC_TRACE){
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n";
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
  }
  
  BOOST_TEST_MESSAGE("inplace    shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);
  
  
}

BOOST_AUTO_TEST_CASE(power_of_5) {

  std::vector<size_t> shape(3,25);
  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  fc::image_stack received(stack);

  fc::inplace_fft_ifft(received);
  
double my_l2norm = l2norm(stack,received);
  const double expected = 1e-1;
  const bool result = my_l2norm<expected;

  if(!result && FC_TRACE){
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n";
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
  }
  
  BOOST_TEST_MESSAGE("inplace    shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);
  
  
}

BOOST_AUTO_TEST_CASE(power_of_7) {

  std::vector<size_t> shape(3,2*7);
  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  fc::image_stack received(stack);

  fc::inplace_fft_ifft(received);
  
double my_l2norm = l2norm(stack,received);
  const double expected = 1e-1;
  const bool result = my_l2norm<expected;

  if(!result && FC_TRACE){
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n";
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
  }
  
  BOOST_TEST_MESSAGE("inplace    shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);
  
  
}

BOOST_AUTO_TEST_CASE(cube_128_shape) {

  std::vector<size_t> shape(3,128);
  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  fc::image_stack received(stack);

  fc::inplace_fft_ifft(received);
  
  double my_l2norm = l2norm(stack,received);
  const double expected = 1e-4;
  const bool result = my_l2norm<expected;

  if(!result && FC_TRACE){
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n";
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
  }
  
  BOOST_TEST_MESSAGE("inplace    shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);
  
  
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(outofplace)

BOOST_AUTO_TEST_CASE(of_prime_shape) {

  std::vector<size_t> shape(3,17);
  shape[fc::row_major::z] = 13;
  shape[fc::row_major::x] = 19;

  fc::image_stack stack(shape);
  fc::image_stack received(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);

  fc::outofplace_fft_ifft(stack, received);
  
double my_l2norm = l2norm(stack,received);

  const double expected = 1e-1;
  const bool result = my_l2norm<expected;
  if(!result && FC_TRACE){
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
  }
  
  BOOST_TEST_MESSAGE("outofplace shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);

}

BOOST_AUTO_TEST_CASE(power_of_2_shape) {

  std::vector<size_t> shape(3,16);

  fc::image_stack stack(shape);
  fc::image_stack received(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);
  
  fc::outofplace_fft_ifft(stack, received);
double my_l2norm = l2norm(stack,received);
  const double expected = 1e-4;
  const bool result = my_l2norm<expected;
  
  if(!result && FC_TRACE){
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
    std::cout << "\nl2norm = " << my_l2norm << "\n";
  }

  BOOST_TEST_MESSAGE("outofplace shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);

}


BOOST_AUTO_TEST_CASE(power_of_3_shape) {

  std::vector<size_t> shape(3,std::pow(3,3));

  fc::image_stack stack(shape);
  fc::image_stack received(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);
  
  fc::outofplace_fft_ifft(stack, received);
double my_l2norm = l2norm(stack,received);  const double expected = 1e-4;
  const bool result = my_l2norm<expected;
  if(!result && FC_TRACE){
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
    std::cout << "\nl2norm = " << my_l2norm << "\n";
  }

  BOOST_TEST_MESSAGE("outofplace shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);

}

BOOST_AUTO_TEST_CASE(power_of_5_shape) {

  std::vector<size_t> shape(3,std::pow(5,2));

  fc::image_stack stack(shape);
  fc::image_stack received(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);
  
  fc::outofplace_fft_ifft(stack, received);
double my_l2norm = l2norm(stack,received);  const double expected = 1e-4;
  const bool result = my_l2norm<expected;
  if(!result && FC_TRACE){
    std::cout << "expected:\n";
    fc::print_stack(stack);
    std::cout << "\n\nreceived:\n";
    fc::print_stack(received);
    std::cout << "\nl2norm = " << my_l2norm << "\n";
  }

    BOOST_TEST_MESSAGE("outofplace shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
    BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);

}

BOOST_AUTO_TEST_CASE(power_of_7_shape) {

  std::vector<size_t> shape(3,14);

  fc::image_stack stack(shape);
  fc::image_stack received(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);
  
  fc::outofplace_fft_ifft(stack, received);
double my_l2norm = l2norm(stack,received);  
  const double expected = 1e-3;
  const bool result = my_l2norm<expected;
  BOOST_TEST_MESSAGE("outofplace shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);

}

BOOST_AUTO_TEST_CASE(cube_128_shape) {

  std::vector<size_t> shape(3,128);

  fc::image_stack stack(shape);
  fc::image_stack received(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);
  
  fc::outofplace_fft_ifft(stack, received);
  double my_l2norm = l2norm(stack,received);  
  const double expected = 1e-3;
  const bool result = my_l2norm<expected;
  BOOST_TEST_MESSAGE("outofplace shape(x,y,z)=" << shape[fc::row_major::x]<< ", " << shape[fc::row_major::y]<< ", " << shape[fc::row_major::z] << "\tl2norm = " << my_l2norm);
  BOOST_REQUIRE_MESSAGE(result,"l2norm = "<< my_l2norm <<" not smaller than " << expected);

}
BOOST_AUTO_TEST_SUITE_END()
