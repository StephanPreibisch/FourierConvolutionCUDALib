FourierConvolutionCUDALib
=========================

Implementation of 3d non-separable convolution using CUDA &amp; FFT Convolution

To compile this code under linux, use the following command (and adjust paths obviously):

nvcc convolution3Dfft.cu --compiler-options '-fPIC' -shared -lcudart -lcufft -I/usr/lib/nvidia-cuda-toolkit/include/ -L/usr/lib/nvidia-cuda-toolkit/lib -lcuda -o libConvolution3D

To compile it under Linux/Mac using NSight, make a new shared library project. Under Project > Properties > NVCC Linker add -lcuda to the command line pattern so that it looks like this:

${COMMAND} ${FLAGS} -lcuda ${OUTPUT_FLAG} ${OUTPUT_PREFIX} ${OUTPUT} ${INPUTS}

Now build the .so library and put it into the Fiji directory.
