#include <cufft.h>
#include <iostream>
#include <fstream>

//#include "../utils.h"

inline static void cudasafe( cudaError_t error, const char* message = 0){
  if(error!=cudaSuccess) { std::cerr << "CUDA ERROR: "<< message << " : " << error << "\n"; exit(1); }
}

inline static void cufftsafe(cufftResult error , const char* message = 0){
  if(error!=CUFFT_SUCCESS) { std::cerr << "CUFFT ERROR: "<< message << " : " << error << "\n"; exit(1); }
}


void cufft_c2c_ptr(cufftComplex* _data, int Nx, int Ny, int Nz){
  
  
  cufftComplex *x_g;

  const size_t N_total = Nx*Ny*Nz;

  cudasafe(cudaMalloc((void **)&x_g,sizeof(cufftComplex) * N_total ));



  // Copy host memory to device
  cudasafe(cudaMemcpy(x_g, _data, sizeof(cufftComplex) * N_total, 
		      cudaMemcpyHostToDevice));
	
  // CUFFT plan
  cufftHandle plan;
  cufftsafe(cufftPlan3d(&plan, Nx,Ny, Nz, CUFFT_C2C)   ,"cufftPlan3d failed");
  cufftsafe(cufftExecC2C(plan, x_g, x_g, CUFFT_FORWARD),"cufftExecC2C(plan, x_g, x_g, CUFFT_FORWARD failed");
  cufftsafe(cufftExecC2C(plan, x_g, x_g, CUFFT_INVERSE),"cufftExecC2C(plan, x_g, x_g, CUFFT_INVERSE failed");


  cudaMemcpy(_data,x_g, sizeof(cufftComplex) * N_total, 
	     cudaMemcpyDeviceToHost);

  //Destroy CUFFT context
  cufftDestroy(plan);
  
  cudaFree(x_g);
    	
  cudaDeviceReset();

  
  for (unsigned int i = 0; i < N_total; ++i)
    {
      _data[i].x /= N_total;
      _data[i].y /= N_total;
    }
  
}


void cufft_c2c(std::vector<cufftComplex>& _data, int Nx, int Ny, int Nz){

  cufft_c2c_ptr(&_data[0],  Nx,  Ny,  Nz);
  
}

double run_cufft_c2c(int Nx, int Ny, int Nz)
{

  cudaSetDevice(0);

  std::cout << "[run_cufft_c2c] Nx = "<< Nx<<" Ny = "<<Ny<<" Nz = "<<Nz<<" \n";

  const long N_total = Nx*Ny*Nz;
  // the host buffer
  cufftComplex *x = (cufftComplex *)malloc(sizeof(cufftComplex) * N_total);
  cufftComplex *y = (cufftComplex *)malloc(sizeof(cufftComplex) * N_total);
  
  for (unsigned int i = 0; i < N_total; ++i)
    {
      x[i].x = 1.*rand()/RAND_MAX;
      x[i].y = 1.*rand()/RAND_MAX;

      y[i] = x[i];
    }

  //transform
  cufft_c2c_ptr(y,Nx,Ny,Nz);
  
  //calc difference

  double diff = 0.;

  for (unsigned int i = 0; i < N_total; ++i)
    {
      diff += (x[i].x-y[i].x)*(x[i].x-y[i].x)+(x[i].y-y[i].y)*(x[i].y-y[i].y);
    }


  // cleanup memory
  free(x);
  free(y);

  return diff;
}
