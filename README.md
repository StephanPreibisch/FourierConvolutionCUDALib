FourierConvolutionCUDALib
=========================

Implementation of 3d non-separable convolution using CUDA &amp; FFT Convolution, originally implemented by Fernando Amat for our Nature Methods paper (http://www.nature.com/nmeth/journal/v11/n6/full/nmeth.2929.html).

Windows Build Instructions
--------------------------

To compile it under Windows, NSight available from the CUDA SDK is suggested. Clone this repository into your cuda-workspace directory. Then make a new shared library project with the same name as the directory.

Under Project > Properties > Build > Settings > Tool Settings > NVCC Linker add -lcufft and -lcuda to the command line pattern so that it looks like this:

${COMMAND} ${FLAGS} -lcufft -lcuda ${OUTPUT_FLAG} ${OUTPUT_PREFIX} ${OUTPUT} ${INPUTS}

Now build the .so/.dll library and put it into the Fiji directory.

This build does not support unit tests.

OSX/Linux Build Instructions
----------------------------

First, make sure that CUDA must be available through `PATH`, `LD_LIBRARY_PATH` or equivalent. The build system is based on cmake, so please install this at any version higher or equal than 2.8.

```bash
$ cd /path/to/repo
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/directory/of/your/choice .. #default is /usr/lib/ or /usr/lib64 depending on your OS
$ make
$ ctest #(optional) in case you'd like to make sure the library does what it should on your system
$ make install
```


Dependencies
------------

The unit tests that come with the library require boost (version 1.42 or higher) to be available to the cmake build system. If cmake is unable to detect them, try:

```
$ cmake -DCMAKE_INSTALL_PREFIX=/directory/of/your/choice -DBOOST_ROOT=/path/to/boost/root .. #default is /usr/lib/ or /usr/lib64 depending on your OS
```

Here, ```/path/to/boost/root``` should contain the boost libraries and the boost headers.


How to get Help
===============

In case something does not work, do not hesitate to open a ticket on our github page: https://github.com/StephanPreibisch/FourierConvolutionCUDALib
