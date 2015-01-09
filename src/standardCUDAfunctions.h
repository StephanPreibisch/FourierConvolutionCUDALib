/*
 * standardCUDAfunctions.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef STANDARDCUDAFUNCTIONS_H_
#define STANDARDCUDAFUNCTIONS_H_

#ifdef _WIN32
#define FUNCTION_PREFIX extern "C" __declspec(dllexport)
#else
#define FUNCTION_PREFIX extern "C"
#endif
//----------------------------------functions to decide whhich GPU to use-------------------------------

FUNCTION_PREFIX int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
FUNCTION_PREFIX int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
FUNCTION_PREFIX int getNumDevicesCUDA();
FUNCTION_PREFIX void getNameDeviceCUDA(int devCUDA, char *name);
FUNCTION_PREFIX long long int getMemDeviceCUDA(int devCUDA);

#endif /* STANDARDCUDAFUNCTIONS_H_ */
