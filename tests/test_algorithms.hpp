#ifndef _TEST_ALGORITHMS_H_
#define _TEST_ALGORITHMS_H_

#include <vector>
#include "image_stack_utils.h"

namespace multiviewnative {

template <typename ImageStackT, typename DimT>
void convolve(const ImageStackT& _image, const ImageStackT& _kernel,
              ImageStackT& _result, const std::vector<DimT>& _offset) {

  if (!_image.num_elements()) return;

  std::vector<DimT> half_kernel(3);
  for (unsigned i = 0; i < 3; ++i) half_kernel[i] = _kernel.shape()[i] / 2;

  float image_value = 0;
  float kernel_value = 0;
  float value = 0;

  for (int image_z = _offset[0]; image_z < int(_image.shape()[0] - _offset[0]);
       ++image_z) {
    for (int image_y = _offset[1];
         image_y < int(_image.shape()[1] - _offset[1]); ++image_y) {
      for (int image_x = _offset[2];
           image_x < int(_image.shape()[2] - _offset[2]); ++image_x) {

        _result[image_z][image_y][image_x] = 0.f;

        image_value = 0;
        kernel_value = 0;
        value = 0;

        // TODO: adapt loops to storage order for the sake of speed (if so,
        // prefetch line to use SIMD)
        for (int kernel_z = 0; kernel_z < int(_kernel.shape()[0]); ++kernel_z) {
          for (int kernel_y = 0; kernel_y < int(_kernel.shape()[1]);
               ++kernel_y) {
            for (int kernel_x = 0; kernel_x < int(_kernel.shape()[2]);
                 ++kernel_x) {

              kernel_value = _kernel[_kernel.shape()[0] - 1 - kernel_z]
                                    [_kernel.shape()[1] - 1 - kernel_y]
                                    [_kernel.shape()[2] - 1 - kernel_x];
              image_value = _image[image_z - half_kernel[0] + kernel_z]
                                  [image_y - half_kernel[1] + kernel_y]
                                  [image_x - half_kernel[2] + kernel_x];

              value += kernel_value * image_value;
            }
          }
        }
        _result[image_z][image_y][image_x] = value;
      }
    }
  }
}

template <typename ImageStackT, typename DimT>
typename ImageStackT::element sum_from_offset(
    const ImageStackT& _image, const std::vector<DimT>& _offset) {

  typename ImageStackT::element value = 0.f;

  // TODO: adapt loops to storage order for the sake of speed (if so, prefetch
  // line to use SIMD)
  for (int image_z = _offset[0]; image_z < int(_image.shape()[0] - _offset[0]);
       ++image_z) {
    for (int image_y = _offset[1];
         image_y < int(_image.shape()[1] - _offset[1]); ++image_y) {
      for (int image_x = _offset[2];
           image_x < int(_image.shape()[2] - _offset[2]); ++image_x) {

        value += _image[image_z][image_y][image_x];
      }
    }
  }

  return value;
}

#ifdef _OPENMP
#include "omp.h"
#endif

template <typename ValueT, typename DimT>
ValueT l2norm(ValueT* _first, ValueT* _second, const DimT& _size) {

  ValueT l2norm = 0.;

#ifdef _OPENMP
  int nthreads = omp_get_num_procs();
  #pragma omp parallel for num_threads(nthreads) shared(l2norm)
#endif
  for (unsigned p = 0; p < _size; ++p)
    l2norm += (_first[p] - _second[p]) * (_first[p] - _second[p]);

  return l2norm;
}

template <typename ImageT, typename OtherT>
float l2norm_within_limits(const ImageT& _first, const OtherT& _second,
                           const float& _rel_lower_limit_per_axis,
                           const float& _rel_upper_limit_per_axis) {

  if (_rel_upper_limit_per_axis < _rel_lower_limit_per_axis) {
    std::cerr << "[l2norm_within_limits]\treceived weird interval ["
              << _rel_lower_limit_per_axis << "," << _rel_upper_limit_per_axis
              << "]\n";
    return 0;
  }

  float l2norm = 0.;

#pragma omp parallel for num_threads(omp_get_num_procs()) shared(l2norm)
  for (int image_z = _first.shape()[0] * _rel_lower_limit_per_axis;
       image_z < int(_first.shape()[0] * _rel_upper_limit_per_axis);
       ++image_z) {
    for (int image_y = _first.shape()[1] * _rel_lower_limit_per_axis;
         image_y < int(_first.shape()[1] * _rel_upper_limit_per_axis);
         ++image_y) {
      for (int image_x = _first.shape()[2] * _rel_lower_limit_per_axis;
           image_x < int(_first.shape()[2] * _rel_upper_limit_per_axis);
           ++image_x) {

        float intermediate = _first[image_z][image_y][image_x] -
                             _second[image_z][image_y][image_x];
        l2norm += (intermediate) * (intermediate);
      }
    }
  }

  return l2norm;
}

template <typename ValueT, typename DimT>
ValueT l1norm(ValueT* _first, ValueT* _second, const DimT& _size) {

  ValueT l1norm = 0.;

#pragma omp parallel for num_threads(omp_get_num_procs()) shared(l1norm)
  for (unsigned p = 0; p < _size; ++p)
    l1norm += std::fabs(_first[p] - _second[p]);

  return l1norm;
}

template <typename ValueT, typename DimT>
ValueT max_value(ValueT* _data, const DimT& _size) {

  ValueT max = *std::max_element(_data, _data + _size);

  return max;
}

template <typename ValueT, typename DimT>
ValueT min_value(ValueT* _data, const DimT& _size) {

  ValueT min = *std::min_element(_data, _data + _size);

  return min;
}
}
#endif /* _TEST_ALGORITHMS_H_ */
