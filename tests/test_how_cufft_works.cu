#define BOOST_TEST_MODULE HOW_CUFFT_WORKS
#include "boost/test/unit_test.hpp"


#include <numeric>
#include <vector>

#include "image_stack_utils.h"

#include "book.h"
#include "cufft.h"

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

template <int value = 42>
struct set_to_float {

  float operator()(){

    return float(value);
    
  }
  
};


struct ramp
{
  size_t value;

  ramp():
    value(0){};
  
  float operator()(){

    return value++;
    
  }
  
};
  

template <typename value_policy = ramp>
struct stack_fixture {

  fc::image_stack stack;
  fc::image_stack kernel;

  template <typename T>
  stack_fixture(const std::vector<T>& _stack_shape,
	       const std::vector<T>& _kernel_shape):
    stack(_stack_shape),
    kernel(_kernel_shape){

    value_policy operation;
    std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
    std::generate(stack.data(),stack.data()+stack.num_elements(),operation);
    
  }
  
};

__global__ void scale(cufftComplex* _array, size_t _size, float _scale){

  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  cufftComplex el;
  if(tid<_size){
    el = _array[tid];
    _array[tid].x = el.x*_scale;
    _array[tid].y = el.y*_scale;
  }
      

}

BOOST_AUTO_TEST_SUITE(fft_ifft)

BOOST_AUTO_TEST_CASE(inplace_returns_equal) {

  std::vector<size_t> shape(3,17);
  shape[fc::row_major::z] = 13;
  shape[fc::row_major::x] = 19;

  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);
  
  std::vector<size_t> shape_for_cufft(shape);
  shape_for_cufft[fc::row_major::x] = (shape[fc::row_major::x]/2) + 1;
  size_t size_for_cufft = std::accumulate(shape_for_cufft.begin(), shape_for_cufft.end(),1,std::multiplies<size_t>());
  
  cufftComplex* d_stack = 0;
  
  HANDLE_ERROR( cudaMalloc( (void**)&(d_stack), size_for_cufft*sizeof(cufftComplex) ) );
  HANDLE_ERROR( cudaMemset( d_stack, 0, size_for_cufft*sizeof(cufftComplex) ));
  HANDLE_ERROR( cudaMemcpy( d_stack, stack.data(), img_size*sizeof(float) , cudaMemcpyHostToDevice ) );

  //FORWARD
  cufftHandle fftPlanFwd;
  cufftPlan3d(&fftPlanFwd, shape[fc::row_major::x], shape[fc::row_major::y], shape[fc::row_major::z], CUFFT_R2C);HANDLE_ERROR_KERNEL;
  cufftExecR2C(fftPlanFwd, (cufftReal*)d_stack, (cufftComplex *)d_stack);HANDLE_ERROR_KERNEL;
  ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;

  //apply scale
  const float scale_ = 1.f/float(img_size);
  unsigned threads = 32;
  unsigned blocks = (size_for_cufft + threads -1) /threads;
  scale<<<blocks,threads>>>(d_stack,size_for_cufft,scale_);
  
  //BACKWARD
  cufftHandle fftPlanInv;
  cufftPlan3d(&fftPlanInv, shape[fc::row_major::x], shape[fc::row_major::y], shape[fc::row_major::z], CUFFT_C2R);HANDLE_ERROR_KERNEL;
  cufftExecC2R(fftPlanInv, (cufftComplex*)d_stack, (cufftReal *)d_stack);HANDLE_ERROR_KERNEL;
  ( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
  
  fc::image_stack received(shape);
  std::fill(received.data(),received.data()+img_size,0);
  HANDLE_ERROR( cudaMemcpy( received.data(), d_stack , img_size*sizeof(float) , cudaMemcpyDeviceToHost ) );

  for(size_t i = 0;i<img_size;++i){
    double diff = received.data()[i]-stack.data()[i];
    BOOST_REQUIRE_MESSAGE(std::abs(diff)<1e-2,"convolved " << received.data()[i] << " not equal expected " << stack.data()[i] << " at " << i );
  }

  HANDLE_ERROR( cudaFree( d_stack));
}

BOOST_AUTO_TEST_CASE(outofplace_returns_equal) {

  std::vector<size_t> shape(3,17);
  shape[fc::row_major::z] = 13;
  shape[fc::row_major::x] = 19;

  fc::image_stack stack(shape);

  for(size_t i = 0;i<stack.num_elements();++i)
    stack.data()[i] = i;

  size_t img_size = std::accumulate(shape.begin(), shape.end(),1,std::multiplies<size_t>());

  BOOST_REQUIRE(img_size > 32);
  
  std::vector<size_t> shape_for_cufft(shape);
  shape_for_cufft[fc::row_major::x] = (shape[fc::row_major::x]/2) + 1;
  size_t size_for_cufft = std::accumulate(shape_for_cufft.begin(), shape_for_cufft.end(),1,std::multiplies<size_t>());
  
  cufftComplex* d_complex = 0;
  cufftReal* d_real = 0;
  
  HANDLE_ERROR( cudaMalloc( (void**)&(d_complex), size_for_cufft*sizeof(cufftComplex) ) );
  HANDLE_ERROR( cudaMemset( d_complex, 0, size_for_cufft*sizeof(cufftComplex) ));

  HANDLE_ERROR( cudaMalloc( (void**)&(d_real), img_size*sizeof(cufftComplex) ) );
  HANDLE_ERROR( cudaMemcpy( d_real, stack.data(), img_size*sizeof(float) , cudaMemcpyHostToDevice ) );

  //FORWARD
  cufftHandle fftPlanFwd;
  cufftPlan3d(&fftPlanFwd, shape[fc::row_major::x], shape[fc::row_major::y], shape[fc::row_major::z], CUFFT_R2C);HANDLE_ERROR_KERNEL;
  cufftExecR2C(fftPlanFwd, d_real, d_complex);HANDLE_ERROR_KERNEL;

  //apply scale
  const float scale_ = 1.f/float(img_size);
  unsigned threads = 32;
  unsigned blocks = (size_for_cufft + threads -1) /threads;
  scale<<<blocks,threads>>>(d_complex,size_for_cufft,scale_);
  
  //BACKWARD
  cufftHandle fftPlanInv;
  cufftPlan3d(&fftPlanInv, shape[fc::row_major::x], shape[fc::row_major::y], shape[fc::row_major::z], CUFFT_C2R);HANDLE_ERROR_KERNEL;
  cufftExecC2R(fftPlanInv, d_complex, d_real);HANDLE_ERROR_KERNEL;

  
  fc::image_stack received(shape);
  std::fill(received.data(),received.data()+img_size,0);
  HANDLE_ERROR( cudaMemcpy( received.data(), d_real , img_size*sizeof(float) , cudaMemcpyDeviceToHost ) );

  for(size_t i = 0;i<img_size;++i){
    double diff = received.data()[i]-stack.data()[i];
    BOOST_REQUIRE_MESSAGE(std::abs(diff)<1e-1,"convolved " << received.data()[i] << " not equal expected " << stack.data()[i] << " at " << i );
  }

  ( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
  ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;

  HANDLE_ERROR( cudaFree( d_real));
  HANDLE_ERROR( cudaFree( d_complex));
}


BOOST_AUTO_TEST_SUITE_END()
