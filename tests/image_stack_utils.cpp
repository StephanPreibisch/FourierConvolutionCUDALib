#define _IMAGE_STACK_UTILS_CPP_
#include <iomanip>
#include "image_stack_utils.h"

namespace multiviewnative {

/**
   \brief function to print an image_stack

   \param[in] _cout input ostream
   \param[in] _marray input multi-dim array, _marray is expected to be defined
   in c_storage_order
   that means, if one has a stack of
   width = 2 (number of columns, x dimension),
   height = 3 (number of rows, y dimension),
   depth = 4 (number of frames, z dimension)
   then this function expects that it can access the array as
   _marray[z][y][x]
   similarly, the dimensions located at _marray.shape() are expected to be
   _marray.shape()[0] = depth
   _marray.shape()[1] = height
   _marray.shape()[2] = height
   \return
   \retval

*/
std::ostream& operator<<(std::ostream& _cout, const image_stack& _marray) {

  if (image_stack::dimensionality != 3) {
    _cout << "dim!=3\n";
    return _cout;
  }

  if (_marray.empty()) {
    _cout << "size=0\n";
    return _cout;
  }

  int precision = _cout.precision();
  _cout << std::setprecision(4);
  const size_t* shapes = _marray.shape();

  _cout << std::setw(9) << "x = ";
  for (size_t x_index = 0; x_index < (shapes[2]); ++x_index) {
    _cout << std::setw(8) << x_index << " ";
  }
  _cout << "\n";
  _cout << std::setfill('-') << std::setw((shapes[2] + 1) * 9) << " "
        << std::setfill(' ') << std::endl;

  for (size_t z_index = 0; z_index < (shapes[0]); ++z_index) {
    _cout << "z[" << std::setw(5) << z_index << "] \n";
    for (size_t y_index = 0; y_index < (shapes[1]); ++y_index) {
      _cout << "y[" << std::setw(5) << y_index << "] ";

      for (size_t x_index = 0; x_index < (shapes[2]); ++x_index) {
        _cout << std::setw(8) << _marray[z_index][y_index][x_index] << " ";
      }

      _cout << "\n";
    }
    _cout << "\n";
  }

  _cout << std::setprecision(precision);
  return _cout;
}
}
