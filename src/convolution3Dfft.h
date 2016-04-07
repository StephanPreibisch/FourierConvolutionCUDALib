#ifndef __CONVOLUTION_3D_FFT_H__
#define __CONVOLUTION_3D_FFT_H__

#ifdef _WIN32
//auto-generated macro file for MSVC libraries on Win7
	#include "FourierConvolutionCUDALib_Export.h"
	#define FUNCTION_PREFIX extern "C" FourierConvolutionCUDALib_EXPORT

#else

	#define FUNCTION_PREFIX extern "C"
	
#endif


//define constants
typedef float imageType;//the kind sof images we are working with (you will need to recompile to work with other types)
static const int MAX_THREADS_CUDA = 1024; //adjust it for your GPU. This is correct for a 2.0 architecture
static const int MAX_BLOCKS_CUDA = 65535;

static const int dimsImage = 3;//so thing can be set at co0mpile time

/*


WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
NOTE: we assume kernel size is the same as image (it has been padded appropriately)
*/
FUNCTION_PREFIX imageType* convolution3DfftCUDA_test(imageType* im,
								int* imDim,
								imageType* kernel,
								int devCUDA);


/*

WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)

TODO: pad data with imSize+kernelSize-1 (kernelSize/2 on each side) to impose the boundary conditions that you want: mirror, zero, etc...). Look at conv2DFFT in the CUDA SDK examples
*/
FUNCTION_PREFIX imageType* convolution3DfftCUDA(imageType* im,
												int* imDim,
												imageType* kernel,
												int* kernelDim,
												int devCUDA);


/*

\brief In-place convolution where image is substituted by the convolution

WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)

TODO: pad data with imSize+kernelSize-1 (kernelSize/2 on each side) to impose the boundary conditions that you want: mirror, zero, etc...). Look at conv2DFFT in the CUDA SDK examples
*/
FUNCTION_PREFIX void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);

/*
\brief Like convolution3DfftCUDAInPlace but we do not pad the kernel with zeros before convolution. We just do nearest neighbor interpolation of the small size kernel FFT. In preliminary tests we do not seem to lose significant precision in the results

NOTE: on a Tesla C2075 the maximum image size seems to be 1024 x  875 x 512 (in float). Approximately 1750 MB. It takes 2.5secs for this maximum size (mostly transferring memory back and forth to GPU)
WARNING: If imSize > 5*kernelSize the maximum relative error in the convolution is 2% (SO WE SACRIFICE PRECISION FOR MEMORY AND LITTLE SPEED UP)
*/
//FUNCTION_PREFIX void convolution3DfftCUDAInPlaceSaveMemory(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);

FUNCTION_PREFIX int selectDeviceWithHighestComputeCapability();
FUNCTION_PREFIX int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
FUNCTION_PREFIX int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
FUNCTION_PREFIX int getNumDevicesCUDA();
FUNCTION_PREFIX void getNameDeviceCUDA(int devCUDA, char *name);
FUNCTION_PREFIX long long int getMemDeviceCUDA(int devCUDA);
FUNCTION_PREFIX int cuda_version();

/**
   \brief function to ask cufft how much memory it'll need for a transform of size shape
   
   \param[in] shape .. dimensions of the signal to transform
   \param[in] len   .. number of dimensions
   
   \return 
   \retval 
   
*/
FUNCTION_PREFIX int gpu_mem_needed_mb(int* shape, int len);


#endif //__CONVOLUTION_3D_FFT_H__
