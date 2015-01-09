FourierConvolutionCUDALib
=========================

Implementation of 3d non-separable convolution using CUDA &amp; FFT Convolution, originally implemented by Fernando Amat for our Nature Methods paper (http://www.nature.com/nmeth/journal/v11/n6/full/nmeth.2929.html).

To compile it under Linux & OS X (cuda must be available through `PATH`, `LD_LIBRARY_PATH` or equivalent):

```bash
$ cd /path/to/repo
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/directory/of/your/choice .. #default is /usr/bin/ or similar
$ make
$ make install
```

NOTE: If you are compiling under Windows, you need to change all 'extern "C"' definitions to 'extern "C" __declspec(dllexport)' for all function calls in the convolution3Dfft.h and convolution3Dfft.cu.
