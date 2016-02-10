#ifndef _TEST_FIXTURES_H_
#define _TEST_FIXTURES_H_
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>

#include <boost/static_assert.hpp>
#include "boost/multi_array.hpp"

// http://www.boost.org/doc/libs/1_55_0/libs/multi_array/doc/user.html
// http://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order
#include "image_stack_utils.h"
#include "test_algorithms.hpp"

namespace fourierconvolution {

template <unsigned short KernelDimSize = 3, unsigned ImageDimSize = 8>
struct convolutionFixture3D {

  static const int n_dim = 3;
  static const unsigned halfKernel = KernelDimSize / 2u;
  static const unsigned imageDimSize = ImageDimSize;
  static const unsigned kernelDimSize = KernelDimSize;

  const unsigned image_size_;
  std::vector<int> image_dims_;
  std::vector<int> padded_image_dims_;
  std::vector<int> asymm_padded_image_dims_;
  image_stack image_;
  image_stack one_;
  image_stack padded_image_;
  image_stack padded_one_;
  image_stack asymm_padded_image_;
  image_stack asymm_padded_one_;
  image_stack image_folded_by_horizontal_;
  image_stack image_folded_by_vertical_;
  image_stack image_folded_by_depth_;
  image_stack image_folded_by_all1_;
  image_stack one_folded_by_asymm_cross_kernel_;
  image_stack one_folded_by_asymm_one_kernel_;
  image_stack one_folded_by_asymm_identity_kernel_;

  const unsigned kernel_size_;
  std::vector<int> kernel_dims_;
  std::vector<int> asymm_kernel_dims_;
  image_stack trivial_kernel_;
  image_stack identity_kernel_;
  image_stack vertical_kernel_;
  image_stack horizont_kernel_;
  image_stack depth_kernel_;
  image_stack all1_kernel_;
  image_stack asymm_cross_kernel_;
  image_stack asymm_one_kernel_;
  image_stack asymm_identity_kernel_;

  std::vector<int> symm_offsets_;
  std::vector<range> symm_ranges_;
  std::vector<int> asymm_offsets_;
  std::vector<range> asymm_ranges_;

  BOOST_STATIC_ASSERT(KernelDimSize % 2 != 0);

 public:
  convolutionFixture3D()
      : image_size_((unsigned)std::pow(ImageDimSize, n_dim)),
        image_dims_(n_dim, ImageDimSize),
        padded_image_dims_(n_dim, ImageDimSize + 2 * (KernelDimSize / 2)),
        asymm_padded_image_dims_(n_dim),
        image_(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        one_(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        padded_image_(boost::extents[ImageDimSize + 2 * (KernelDimSize / 2)]
                                    [ImageDimSize + 2 * (KernelDimSize / 2)]
                                    [ImageDimSize + 2 * (KernelDimSize / 2)]),
        padded_one_(boost::extents[ImageDimSize + 2 * (KernelDimSize / 2)]
                                  [ImageDimSize + 2 * (KernelDimSize / 2)]
                                  [ImageDimSize + 2 * (KernelDimSize / 2)]),
        asymm_padded_image_(),
        asymm_padded_one_(),
        image_folded_by_horizontal_(
            boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        image_folded_by_vertical_(
            boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        image_folded_by_depth_(
            boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        image_folded_by_all1_(
            boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        one_folded_by_asymm_cross_kernel_(
            boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        one_folded_by_asymm_one_kernel_(
            boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        one_folded_by_asymm_identity_kernel_(
            boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
        kernel_size_((unsigned)std::pow(KernelDimSize, n_dim)),
        kernel_dims_(n_dim, KernelDimSize),
        asymm_kernel_dims_(n_dim, KernelDimSize),
        trivial_kernel_(
            boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
        identity_kernel_(
            boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
        vertical_kernel_(
            boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
        horizont_kernel_(
            boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
        depth_kernel_(
            boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
        all1_kernel_(
            boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
        asymm_cross_kernel_(boost::extents[KernelDimSize + 1][KernelDimSize]
                                          [KernelDimSize - 1]),
        asymm_one_kernel_(boost::extents[KernelDimSize + 1][KernelDimSize]
                                        [KernelDimSize - 1]),
        asymm_identity_kernel_(boost::extents[KernelDimSize + 1][KernelDimSize]
                                             [KernelDimSize - 1]),
        symm_offsets_(n_dim, halfKernel),
        symm_ranges_(n_dim, range(halfKernel, halfKernel + ImageDimSize)),
        asymm_offsets_(n_dim, 0),
        asymm_ranges_(n_dim) {

    // FILL KERNELS

    std::fill(trivial_kernel_.data(), trivial_kernel_.data() + kernel_size_,
              0.f);
    std::fill(identity_kernel_.data(), identity_kernel_.data() + kernel_size_,
              0.f);
    std::fill(vertical_kernel_.data(), vertical_kernel_.data() + kernel_size_,
              0.f);
    std::fill(depth_kernel_.data(), depth_kernel_.data() + kernel_size_, 0.f);
    std::fill(horizont_kernel_.data(), horizont_kernel_.data() + kernel_size_,
              0.f);
    std::fill(all1_kernel_.data(), all1_kernel_.data() + kernel_size_, 1.f);
    std::fill(asymm_cross_kernel_.data(),
              asymm_cross_kernel_.data() + asymm_cross_kernel_.num_elements(),
              0.f);
    std::fill(asymm_one_kernel_.data(),
              asymm_one_kernel_.data() + asymm_one_kernel_.num_elements(), 0.f);
    std::fill(
        asymm_identity_kernel_.data(),
        asymm_identity_kernel_.data() + asymm_identity_kernel_.num_elements(),
        0.f);

    identity_kernel_.data()[kernel_size_ / 2] = 1.;

    for (unsigned int index = 0; index < KernelDimSize; ++index) {
      horizont_kernel_[halfKernel][halfKernel][index] = float(index + 1);
      vertical_kernel_[halfKernel][index][halfKernel] = float(index + 1);
      depth_kernel_[index][halfKernel][halfKernel] = float(index + 1);
    }

    asymm_identity_kernel_[asymm_cross_kernel_.shape()[0] / 2]
                          [asymm_cross_kernel_.shape()[1] / 2]
                          [asymm_cross_kernel_.shape()[2] / 2] = 1.f;

    for (int z_index = 0; z_index < int(asymm_cross_kernel_.shape()[0]);
         ++z_index) {
      for (int y_index = 0; y_index < int(asymm_cross_kernel_.shape()[1]);
           ++y_index) {
        for (int x_index = 0; x_index < int(asymm_cross_kernel_.shape()[2]);
             ++x_index) {

          if (z_index == (int)asymm_cross_kernel_.shape()[0] / 2 &&
              y_index == (int)asymm_cross_kernel_.shape()[1] / 2) {
            asymm_cross_kernel_[z_index][y_index][x_index] = x_index + 1;
            asymm_one_kernel_[z_index][y_index][x_index] = 1;
          }

          if (x_index == (int)asymm_cross_kernel_.shape()[2] / 2 &&
              y_index == (int)asymm_cross_kernel_.shape()[1] / 2) {
            asymm_cross_kernel_[z_index][y_index][x_index] = z_index + 101;
            asymm_one_kernel_[z_index][y_index][x_index] = 1;
          }

          if (x_index == (int)asymm_cross_kernel_.shape()[2] / 2 &&
              z_index == (int)asymm_cross_kernel_.shape()[0] / 2) {
            asymm_cross_kernel_[z_index][y_index][x_index] = y_index + 11;
            asymm_one_kernel_[z_index][y_index][x_index] = 1;
          }
        }
      }
    }

    // FILL IMAGES
    unsigned padded_image_axis = ImageDimSize + 2 * halfKernel;
    unsigned padded_image_size = std::pow(padded_image_axis, n_dim);
    std::fill(image_.data(), image_.data() + image_size_, 0.f);
    std::fill(one_.data(), one_.data() + image_size_, 0.f);
    std::fill(padded_image_.data(), padded_image_.data() + padded_image_size,
              0.f);
    std::fill(padded_one_.data(), padded_one_.data() + padded_image_size, 0.f);
    padded_one_[padded_image_axis / 2][padded_image_axis / 2]
               [padded_image_axis / 2] = 1.f;
    one_[ImageDimSize / 2][ImageDimSize / 2][ImageDimSize / 2] = 1.f;

    unsigned image_index = 0;
    for (int z_index = 0; z_index < int(image_dims_[0]); ++z_index) {
      for (int y_index = 0; y_index < int(image_dims_[1]); ++y_index) {
        for (int x_index = 0; x_index < int(image_dims_[2]); ++x_index) {
          image_[z_index][y_index][x_index] = float(image_index++);
        }
      }
    }

    // PADD THE IMAGE FOR CONVOLUTION
    range axis_subrange = range(halfKernel, halfKernel + ImageDimSize);
    image_stack_view padded_image_original = padded_image_
        [boost::indices[axis_subrange][axis_subrange][axis_subrange]];
    padded_image_original = image_;

    image_stack padded_image_folded_by_horizontal = padded_image_;
    image_stack padded_image_folded_by_vertical = padded_image_;
    image_stack padded_image_folded_by_depth = padded_image_;
    image_stack padded_image_folded_by_all1 = padded_image_;

    // PREPARE ASYMM IMAGES
    // std::vector<unsigned> asymm_padded_image_shape(n_dim);

    std::copy(asymm_cross_kernel_.shape(),
              asymm_cross_kernel_.shape() + image_stack::dimensionality,
              asymm_kernel_dims_.begin());


    for (int i = 0; i < n_dim; ++i) {
      asymm_offsets_[i] = asymm_cross_kernel_.shape()[i]/2;
      asymm_ranges_[i] =
          range(asymm_offsets_[i], asymm_offsets_[i] + ImageDimSize);
      asymm_padded_image_dims_[i] = ImageDimSize + 2 * asymm_offsets_[i];
    }

    asymm_padded_image_.resize(asymm_padded_image_dims_);
    asymm_padded_one_.resize(asymm_padded_image_dims_);
    std::fill(asymm_padded_image_.data(),
              asymm_padded_image_.data() + asymm_padded_image_.num_elements(),
              0.f);
    std::fill(asymm_padded_one_.data(),
              asymm_padded_one_.data() + asymm_padded_one_.num_elements(), 0.f);
    asymm_padded_one_[asymm_padded_one_.shape()[0] / 2]
                     [asymm_padded_one_.shape()[1] / 2]
                     [asymm_padded_one_.shape()[2] / 2] = 1.f;

    image_stack_view asymm_padded_image_original = asymm_padded_image_
        [boost::indices[asymm_ranges_[0]][asymm_ranges_[1]][asymm_ranges_[2]]];
    asymm_padded_image_original = image_;

    image_stack asymm_padded_one_folded_by_asymm_cross_kernel =
        asymm_padded_one_;
    image_stack asymm_padded_one_folded_by_asymm_one_kernel = asymm_padded_one_;
    image_stack asymm_padded_one_folded_by_asymm_identity_kernel =
        asymm_padded_one_;

    // CONVOLVE
    convolve(padded_image_, horizont_kernel_, padded_image_folded_by_horizontal,
             symm_offsets_);
    convolve(padded_image_, vertical_kernel_, padded_image_folded_by_vertical,
             symm_offsets_);
    convolve(padded_image_, depth_kernel_, padded_image_folded_by_depth,
             symm_offsets_);
    convolve(padded_image_, all1_kernel_, padded_image_folded_by_all1,
             symm_offsets_);

    convolve(asymm_padded_one_, asymm_cross_kernel_,
             asymm_padded_one_folded_by_asymm_cross_kernel, asymm_offsets_);
    convolve(asymm_padded_one_, asymm_one_kernel_,
             asymm_padded_one_folded_by_asymm_one_kernel, asymm_offsets_);
    convolve(asymm_padded_one_, asymm_identity_kernel_,
             asymm_padded_one_folded_by_asymm_identity_kernel, asymm_offsets_);

    // EXTRACT NON-PADDED CONTENT FROM CONVOLVED IMAGE STACKS
    image_folded_by_horizontal_ = padded_image_folded_by_horizontal
        [boost::indices[axis_subrange][axis_subrange][axis_subrange]];
    image_folded_by_vertical_ = padded_image_folded_by_vertical
        [boost::indices[axis_subrange][axis_subrange][axis_subrange]];
    image_folded_by_depth_ = padded_image_folded_by_depth
        [boost::indices[axis_subrange][axis_subrange][axis_subrange]];
    image_folded_by_all1_ = padded_image_folded_by_all1
        [boost::indices[axis_subrange][axis_subrange][axis_subrange]];

    one_folded_by_asymm_cross_kernel_ =
        asymm_padded_one_folded_by_asymm_cross_kernel
            [boost::indices[asymm_ranges_[0]][asymm_ranges_[1]]
                           [asymm_ranges_[2]]];
    one_folded_by_asymm_one_kernel_ =
        asymm_padded_one_folded_by_asymm_one_kernel
            [boost::indices[asymm_ranges_[0]][asymm_ranges_[1]]
                           [asymm_ranges_[2]]];
    one_folded_by_asymm_identity_kernel_ =
        asymm_padded_one_folded_by_asymm_identity_kernel
            [boost::indices[asymm_ranges_[0]][asymm_ranges_[1]]
                           [asymm_ranges_[2]]];
  }

  virtual ~convolutionFixture3D() {};

  static const unsigned image_axis_size = ImageDimSize;
  static const unsigned kernel_axis_size = KernelDimSize;
};

typedef convolutionFixture3D<> default_3D_fixture;

}  // namespace

#endif
