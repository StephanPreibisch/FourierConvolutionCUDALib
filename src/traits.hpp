#ifndef _TRAITS_H_
#define _TRAITS_H_

namespace fourierconvolution {

struct row_major {

    const static size_t x = 2;
    const static size_t y = 1;
    const static size_t z = 0;

    const static size_t w = x;
    const static size_t h = y;
    const static size_t d = z;

    const static size_t in_x = x;
    const static size_t in_y = y;
    const static size_t in_z = z;

  };

}
#endif /* _TRAITS_H_ */
