#ifndef _IMAGE_STACK_UTILS_H_
#define _IMAGE_STACK_UTILS_H_
#include <vector>
#include <iostream>
#include <iomanip>
#include "boost/multi_array.hpp"

namespace fourierconvolution {

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

  template <typename stack_type>
  void print_stack(const stack_type& _marray) {

    if (stack_type::dimensionality != 3) {
      std::cout << "dim!=3\n";
      return;
    }

    if (_marray.empty()) {
      std::cout << "size=0\n";
      return;
    }

    int precision = std::cout.precision();
    std::cout << std::setprecision(4);
    const size_t* shapes = _marray.shape();

    std::cout << std::setw(9) << "x = ";
    for (size_t x_index = 0; x_index < (shapes[2]); ++x_index) {
      std::cout << std::setw(8) << x_index << " ";
    }
    std::cout << "\n";
    std::cout << std::setfill('-') << std::setw((shapes[2] + 1) * 9) << " "
	      << std::setfill(' ') << std::endl;

    for (size_t z_index = 0; z_index < (shapes[0]); ++z_index) {
      std::cout << "z[" << std::setw(5) << z_index << "] \n";
      for (size_t y_index = 0; y_index < (shapes[1]); ++y_index) {
        std::cout << "y[" << std::setw(5) << y_index << "] ";

        for (size_t x_index = 0; x_index < (shapes[2]); ++x_index) {
          std::cout << std::setw(8) << _marray[z_index][y_index][x_index] << " ";
        }

        std::cout << "\n";
      }
      std::cout << "\n";
    }

    std::cout << std::setprecision(precision);
    return;
  }

    template <typename stack_type>
    std::string stack_to_string(const stack_type& _marray) {

      std::ostringstream msg;
      if (stack_type::dimensionality != 3) {
	msg << "dim!=3\n";
	return msg.str();
      }

      if (_marray.empty()) {
	msg << "size=0\n";
	return msg.str();
      }

      int precision = msg.precision();
      msg << std::setprecision(8);
      const size_t* shapes = _marray.shape();

      //slice out in z
      std::vector<size_t> z_indices;
      if(shapes[0]>16){
	for (size_t i = 0; i < (3); ++i) {
	  z_indices.push_back(i);
	}

	for (size_t i = (shapes[0]/2)-1; i <= ((shapes[0]/2)+1); ++i) {
	  z_indices.push_back(i);
	}

	for (size_t i = (shapes[0]-3); i < (shapes[0]); ++i) {
	  z_indices.push_back(i);
	}
      } else {
	z_indices.resize(16);
	for (size_t i = 0; i < z_indices.size(); ++i) {
	  z_indices[i] = i;
	}
      }

      std::vector<size_t> y_indices;
      if(shapes[1]>16){
		for (size_t i = 0; i < (3); ++i) {
	  y_indices.push_back(i);
	}

	for (size_t i = (shapes[0]/2)-1; i <= ((shapes[0]/2)+1); ++i) {
	  y_indices.push_back(i);
	}

	for (size_t i = (shapes[0]-3); i < (shapes[0]); ++i) {
	  y_indices.push_back(i);
	}
      } else {
	y_indices.resize(16);
	for (size_t i = 0; i < y_indices.size(); ++i) {
	  y_indices[i] = i;
	}
      }

      std::vector<size_t> x_indices;
      if(shapes[0]>16){
	for (size_t i = 0; i < (3); ++i) {
	  x_indices.push_back(i);
	}

	for (size_t i = (shapes[0]/2)-1; i <= ((shapes[0]/2)+1); ++i) {
	  x_indices.push_back(i);
	}

	for (size_t i = (shapes[0]-3); i < (shapes[0]); ++i) {
	  x_indices.push_back(i);
	}
      } else {
	x_indices.resize(16);
	for (size_t i = 0; i < x_indices.size(); ++i) {
	  x_indices[i] = i;
	}
      }

      //draw header
      const int col_width = 13;
      msg << std::setw(col_width) << "x = ";      
      if(shapes[2]<=16){
	for (size_t x_index = 0; x_index < (shapes[2]); ++x_index) {
	  msg << std::setw(col_width-1) << x_index << " ";
	}
      } else {
	for (size_t x_index = 0; x_index < (3); ++x_index) {
	  msg << std::setw(col_width-1) << x_index << " ";
	}
	msg << std::setw(9) << "...";
	for (size_t x_index = (shapes[2]/2)-1; x_index <= ((shapes[2]/2)+1); ++x_index) {
	  msg << std::setw(col_width-1) << x_index << " ";
	}
	msg << std::setw(9) << "...";
	for (size_t x_index = (shapes[2])-2; x_index < (shapes[2]); ++x_index) {
	  msg << std::setw(col_width-1) << x_index << " ";
	}
      }

      
      //draw stack
      msg << "\n";
      if(shapes[2]<=16){      
	msg << std::setfill('-') << std::setw((shapes[2] + 1) * 9);
      } else {
	msg << std::setfill('-') << std::setw(10*9);
      }

      msg << " " << std::setfill(' ') << std::endl;

      for (size_t z = 0; z < (z_indices.size()); ++z) {
	size_t z_index = z_indices[z];
	msg << "z[" << std::setw(5) << z_index << "] \n";
	
	for (size_t y = 0; y < (y_indices.size()); ++y) {
	  size_t y_index = y_indices[y];
	  msg << "y[" << std::setw(5) << y_index << "] ";

	  if(shapes[2]<=16){
	    for (size_t x_index = 0; x_index < (shapes[2]); ++x_index) {
	      msg << std::setw(col_width-1) << _marray[z_index][y_index][x_index] << " ";
	    }
	  } else {
	    for (size_t x = 0; x < x_indices.size(); ++x) {
	      if(x % 3 == 0)
		msg << std::setw(col_width) << "...";
	      msg << std::setw(col_width-1) << _marray[z_index][y_index][x_indices[x]] << " ";
	    }
	    
	  }
	  msg << "\n";
	}
	msg << "\n";
      }

      msg << std::setprecision(precision);
      return msg.str();
      
    }



}
#endif /* _IMAGE_STACK_UTILS_H_ */
