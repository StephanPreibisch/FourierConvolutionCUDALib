FourierConvolutionCUDALib
=========================

Implementation of 3d non-separable convolution using CUDA &amp; FFT Convolution, originally implemented by Fernando Amat for our Nature Methods
paper (http://www.nature.com/nmeth/journal/v11/n6/full/nmeth.2929.html).

To compile this code under linux, use the following command (and adjust paths obviously):

nvcc convolution3Dfft.cu --compiler-options '-fPIC' -shared -lcudart -lcufft -I/usr/lib/nvidia-cuda-toolkit/include/ -L/usr/lib/nvidia-cuda-toolkit/lib -lcuda -o libConvolution3D

To compile it under Linux/Mac using NSight, make a new shared library project. Under Project > Properties > NVCC Linker add -lcuda and -lcufft
to the command line pattern so that it looks like this:

${COMMAND} ${FLAGS} -lcufft -lcuda ${OUTPUT_FLAG} ${OUTPUT_PREFIX} ${OUTPUT} ${INPUTS}

Now build the .so library and put it into the Fiji directory.
