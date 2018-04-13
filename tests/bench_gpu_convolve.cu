#include <boost/timer/timer.hpp>

#include "padd_utils.h"

#include "convolution3Dfft.h"
#include "test_utils.hpp"
#include "image_stack_utils.h"
#include "traits.hpp"

#include <vector>
#include <iostream>

using namespace boost::timer;

int main(int argc, char** argv) {


  std::vector<int> image_dims(3,64);
  std::size_t image_len = std::pow(64,3);
  std::vector<float> image(image_len,0.);

  std::vector<int> kernel_dims(3,3);
  std::size_t kernel_len = std::pow(3,3);
  std::vector<float> kernel(kernel_len,0);

  cpu_timer timer;
  for (int i = 0;i<10;++i){

    convolution3DfftCUDAInPlace(&image[0], &image_dims[0] ,
                                &kernel[0], &kernel_dims[0] ,
                                selectDeviceWithHighestComputeCapability());
  }
  std::cout << "inplace, 10x, (image 64**3, kernel 3**3)" << timer.format() << '\n';


}
