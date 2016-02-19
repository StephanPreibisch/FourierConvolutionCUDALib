#include <cufft.h>
#include <iostream>
#include <fstream>

//#include "../utils.h"

inline static void cudasafe( cudaError_t error, char* message = 0){
  if(error!=cudaSuccess) { std::cerr << "ERROR: "<< message << " : " << error << "\n"; exit(1); }
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


  printf("Nx = %ld Ny = %ld Nz = %ld \n",Nx,Ny, Nz);

  const long N_total = Nx*Ny*Nz;
  // the host buffer
  cufftComplex *x = (cufftComplex *)malloc(sizeof(cufftComplex) * N_total);
  cufftComplex *y = (cufftComplex *)malloc(sizeof(cufftComplex) * N_total);
  
  cufftComplex *x_g;

  cudasafe(cudaMalloc((void **)&x_g,sizeof(cufftComplex) * N_total ));

  for (unsigned int i = 0; i < N_total; ++i)
    {
      x[i].x = 1.*rand()/RAND_MAX;
      x[i].y = 1.*rand()/RAND_MAX;

    }

	

  // Copy host memory to device
  cudasafe(cudaMemcpy(x_g, x, sizeof(cufftComplex) * N_total, 
		      cudaMemcpyHostToDevice));
	
  // CUFFT plan
  cufftHandle plan;
  cufftPlan3d(&plan, Nx,Ny, Nz, CUFFT_C2C);

  cufftExecC2C(plan, x_g, x_g, CUFFT_FORWARD);

  cufftExecC2C(plan, x_g, x_g, CUFFT_INVERSE);


  cudaMemcpy(y,x_g, sizeof(cufftComplex) * N_total, 
	     cudaMemcpyDeviceToHost);

  //calc difference

  double diff = 0.;

  for (unsigned int i = 0; i < N_total; ++i)
    {
      y[i].x *= 1./N_total;
      y[i].y *= 1./N_total;
	  
      diff += (x[i].x-y[i].x)*(x[i].x-y[i].x)+(x[i].y-y[i].y)*(x[i].y-y[i].y);
    }

  printf("\nfirst element x: \t %f + %fj\n",x[0].x,x[0].y);
  printf("\nfirst element y: \t %f + %fj\n",y[0].x,y[0].y);

  printf("\nL2 difference: \t %f\n",diff);
  
  //Destroy CUFFT context
  cufftDestroy(plan);

  // cleanup memory
  free(x);
  free(y);
  
  cudaFree(x_g);
    	
  cudaDeviceReset();
  exit(0);
}
