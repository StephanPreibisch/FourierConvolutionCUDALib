/*
 * standardCUDAfunctions.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef STANDARDCUDAFUNCTIONS_H_
#define STANDARDCUDAFUNCTIONS_H_

#ifdef _WIN32
//auto-generated macro file for MSVC libraries on Win7
	#include "convolution3Dfft_Export.h"
	#define FUNCTION_PREFIX extern "C" FourierConvolutionCUDALib_EXPORT
#else
	#define FUNCTION_PREFIX extern "C"
#endif
//----------------------------------functions to decide whhich GPU to use-------------------------------

FUNCTION_PREFIX int selectDeviceWithHighestComputeCapability();
FUNCTION_PREFIX int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
FUNCTION_PREFIX int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
FUNCTION_PREFIX int getNumDevicesCUDA();
FUNCTION_PREFIX void getNameDeviceCUDA(int devCUDA, char *name);
FUNCTION_PREFIX long long int getMemDeviceCUDA(int devCUDA);

#endif /* STANDARDCUDAFUNCTIONS_H_ */
