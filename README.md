FourierConvolutionCUDALib
=========================

Implementation of 3d non-separable convolution using CUDA &amp; FFT Convolution, originally implemented by Fernando Amat for our Nature Methods paper (http://www.nature.com/nmeth/journal/v11/n6/full/nmeth.2929.html).

Windows Build Instructions
--------------------------

To compile it under Windows, NSight available from the CUDA SDK is suggested. Clone this repository into your cuda-workspace directory. Then make a new shared library project with the same name as the directory.

Under Project > Properties > Build > Settings > Tool Settings > NVCC Linker add -lcufft and -lcuda to the command line pattern so that it looks like this:

${COMMAND} ${FLAGS} -lcufft -lcuda ${OUTPUT_FLAG} ${OUTPUT_PREFIX} ${OUTPUT} ${INPUTS}

Now build the .so/.dll library and put it into the Fiji directory.

OSX/Linux Build Instructions
----------------------------

First, make sure that CUDA must be available through `PATH`, `LD_LIBRARY_PATH` or equivalent. The build system is based on cmake, so please install this at any version higher or equal than 2.8.

```bash
$ cd /path/to/repo
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/directory/of/your/choice .. #default is /usr/bin/ or similar
$ make
$ make install
```


