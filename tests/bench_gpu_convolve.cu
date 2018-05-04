#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>

#include "padd_utils.h"

#include "convolution3Dfft.h"
#include "test_utils.hpp"
#include "image_stack_utils.h"
#include "traits.hpp"

#include <vector>
#include <iostream>

using namespace boost::timer;
namespace po = boost::program_options;

int main(int ac, char** av) {

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("image_size", po::value<int>()->default_value(128), "set the 3D image size, so the image will extent sizexsizexsize")
    ("kernel_size", po::value<int>()->default_value(3), "set the kernel size, so the kernel will extent sizexsizexsize")
    ("gpu", po::value<int>()->default_value(-1), "gpu device to use, if value=-1, the highest device with highest compute capability is used")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  int device_id = vm["gpu"].as<int>();
  if(device_id < 0)
    device_id = selectDeviceWithHighestComputeCapability();

  std::vector<int> image_dims(3,vm["image_size"].as<int>());
  std::size_t image_len = std::pow(vm["image_size"].as<int>(),3);
  std::vector<float> image(image_len,0.);

  std::vector<int> kernel_dims(3,vm["kernel_size"].as<int>());
  std::size_t kernel_len = std::pow(vm["kernel_size"].as<int>(),3);
  std::vector<float> kernel(kernel_len,0);

  cpu_timer timer;
  for (int i = 0;i<10;++i){

    convolution3DfftCUDAInPlace(&image[0], &image_dims[0] ,
                                &kernel[0], &kernel_dims[0] ,
                                device_id);
  }
  std::cout << "[gpu "<< device_id << "] inplace, 10x, (image "<< image_dims.front() <<"**3, kernel "<< kernel_dims.front() <<"**3)" << timer.format() << '\n';


}
