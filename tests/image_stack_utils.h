#ifndef _IMAGE_STACK_UTILS_H_
#define _IMAGE_STACK_UTILS_H_
#include <vector>
#include <iostream>
#include "boost/multi_array.hpp"

namespace multiviewnative {

  typedef boost::multi_array<float, 3> image_stack;
  typedef boost::multi_array_ref<float, 3> image_stack_ref;
  typedef boost::const_multi_array_ref<float, 3> image_stack_cref;
  typedef boost::multi_array<float, 2> image_frame;
  typedef boost::multi_array_ref<float, 2> image_frame_ref;
  typedef boost::const_multi_array_ref<float, 2> image_frame_cref;
  typedef image_stack::array_view<3>::type image_stack_view;
  typedef image_stack::array_view<2>::type image_stack_frame;
  typedef image_stack::array_view<1>::type image_stack_line;
  typedef boost::multi_array_types::index_range range;
  typedef boost::general_storage_order<3> storage;

  std::ostream& operator<<(std::ostream&, const image_stack&);

  template <typename DimT, typename ODimT>
  void adapt_extents_for_fftw_inplace(const DimT& _extent, ODimT& _value,
				      const storage& _storage =
				      boost::c_storage_order()) {

    std::fill(_value.begin(), _value.end(), 0);

    std::vector<int> storage_order(_extent.size());
    for (size_t i = 0; i < _extent.size(); ++i)
      storage_order[i] = _storage.ordering(i);

    size_t lowest_storage_index =
      std::min_element(storage_order.begin(), storage_order.end()) -
      storage_order.begin();

    for (size_t i = 0; i < _extent.size(); ++i)
      _value[i] =
        (lowest_storage_index == i) ? 2 * (_extent[i] / 2 + 1) : _extent[i];
  }

  template <typename StorageT, typename DimT, typename ODimT>
  void adapt_shape_for_fftw_inplace(const DimT& _extent,
				    ODimT& _value,
				    const StorageT& _storage) {

    std::fill(_value.begin(), _value.end(), 0);

    size_t lowest_storage_index =
      std::min_element(_storage.begin(), _storage.end()) - _storage.begin();

    for (size_t i = 0; i < _extent.size(); ++i)
      _value[i] =
        (lowest_storage_index == i) ? 2 * (_extent[i] / 2 + 1) : _extent[i];
  }



template <typename DimT, typename ODimT>
void adapt_extents_for_cufft_inplace(const DimT& _extent, ODimT& _value,
				     const storage& _storage =
				     boost::c_storage_order()) {

  adapt_extents_for_fftw_inplace(_extent, _value, _storage);
}

template <typename StorageT, typename DimT, typename ODimT>
void adapt_shape_for_cufft_inplace(const DimT& _extent,
				   ODimT& _value,
				   const StorageT& _storage) {

  adapt_shape_for_fftw_inplace(_extent, _value, _storage);
}

}
#endif /* _IMAGE_STACK_UTILS_H_ */
