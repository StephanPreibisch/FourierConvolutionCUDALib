/*

*/


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cufft.h>
#include <iostream>
#include <fstream>

#include "../utils.h"

void cudasafe( cudaError_t error, char* message = 0){
  if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

void runTest(int argc, char **argv);

int main(int argc, char **argv)
{
    runTest(argc, argv);
}

void runTest(int argc, char **argv)
{

  cudaSetDevice(0);

  long Nx = 256;
  long Ny = 256;
  long Nz = 256;
  
  if (argc==1){
	std::cout<<"usage: cufft_test Nx Ny Nz"<<std::endl;
    return;
  }
  
  if (argc>1)
    Nx = atol(argv[1]);
  
  if (argc>2)
    Ny  = atol(argv[2]);

  if (argc>3)
    Nz  = atol(argv[2]);

  const long Nz_half = Nz/2+1;

  printf("Nx = %ld Ny = %ld Nz = %ld \n",Nx,Ny, Nz);

  const long N_total = Nx*Ny*Nz;
  const long N_total_half = Nx*Ny*Nz_half;

  // the host buffer
  cufftReal *x = (cufftReal *)malloc(sizeof(cufftReal) * N_total);
  cufftReal *y = (cufftReal *)malloc(sizeof(cufftReal) * N_total);
  
  cufftComplex *y_g;
  cufftReal *x_g;

  cudasafe(cudaMalloc((void **)&x_g,sizeof(cufftReal) * N_total));
  cudasafe(cudaMalloc((void **)&y_g,sizeof(cufftComplex) * N_total_half));

  for (unsigned int i = 0; i < N_total; ++i)
    {
	  x[i] = 1.*rand()/RAND_MAX;
	}

	

  // Copy host memory to device
  cudasafe(cudaMemcpy(x_g, x, sizeof(cufftReal) * N_total, 
					  cudaMemcpyHostToDevice));
	
  // CUFFT plan
  cufftHandle plan_fwd, plan_bwd;
  cufftPlan3d(&plan_fwd, Nx,Ny, Nz, CUFFT_R2C);

  cufftPlan3d(&plan_bwd, Nx,Ny, Nz, CUFFT_C2R);

  cufftExecR2C(plan_fwd, x_g, y_g);

  cufftExecC2R(plan_bwd, y_g, x_g);


  cudaMemcpy(y,x_g, sizeof(cufftReal) * N_total, 
			 cudaMemcpyDeviceToHost);

  //calc difference

  double diff = 0.;

  for (unsigned int i = 0; i < N_total; ++i)
	{
	  y[i] *= 1./N_total;
	  
	  diff += (x[i]-y[i])*(x[i]-y[i]);
	}

  printf("\nfirst element x: \t %f\n",x[0]);
  printf("\nfirst element y: \t %f\n",y[0]);

  printf("\nL2 difference: \t %f\n",diff);
  
  //Destroy CUFFT context
  cufftDestroy(plan_fwd);
  cufftDestroy(plan_bwd);

  // cleanup memory
  free(x);
  free(y);
  
  cudaFree(x_g);
    	
  cudaDeviceReset();
  exit(0);
}
