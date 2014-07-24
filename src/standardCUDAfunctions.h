/*
 * standardCUDAfunctions.h
 *
 *  Created on: Jul 24, 2014
 *      Author: preibisch
 */

#ifndef STANDARDCUDAFUNCTIONS_H_
#define STANDARDCUDAFUNCTIONS_H_

//----------------------------------functions to decide whhich GPU to use-------------------------------

extern "C" int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
extern "C" int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
extern "C" int getNumDevicesCUDA();
extern "C" void getNameDeviceCUDA(int devCUDA, char *name);
extern "C" long long int getMemDeviceCUDA(int devCUDA);

#endif /* STANDARDCUDAFUNCTIONS_H_ */
