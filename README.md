FourierConvolutionCUDALib
=========================

Implementation of 3d non-separable convolution using CUDA &amp; FFT Convolution, originally implemented by Fernando Amat for our Nature Methods paper (http://www.nature.com/nmeth/journal/v11/n6/full/nmeth.2929.html).

To compile it under Linux/Mac/Windows I suggest NSight. Clone this repository into your cuda-workspace directory. Then make a new shared library project with the same name as the directory.

Under Project > Properties > Build > Settings > Tool Settings > NVCC Linker add -lcufft and -lcuda to the command line pattern so that it looks like this:

${COMMAND} ${FLAGS} -lcufft -lcuda ${OUTPUT_FLAG} ${OUTPUT_PREFIX} ${OUTPUT} ${INPUTS}

Now build the .so/.dll library and put it into the Fiji directory.

