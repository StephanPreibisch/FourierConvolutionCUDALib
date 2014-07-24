#include "convolution3Dfft.h"
#include "book.h"
#include "cuda.h"
#include "cufft.h"
#include <iostream>
#include <math.h>


__device__ static const float PI_2 = 6.28318530717958620f;
__device__ static const float PI_1 =  3.14159265358979310f;

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
//Adapted from CUDA SDK examples
__device__ void mulAndScale(cufftComplex& a, const cufftComplex& b, const float& c)
{
    cufftComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a = t;
};

__global__ void __launch_bounds__(MAX_THREADS_CUDA)  modulateAndNormalize_kernel(cufftComplex *d_Dst, cufftComplex *d_Src, long long int dataSize,float c)
{
    long long int i = (long long int)blockDim.x * (long long int)blockIdx.x + (long long int)threadIdx.x;
	long long int offset = (long long int)blockDim.x * (long long int)gridDim.x;
    while(i < dataSize)
	{		

		cufftComplex a = d_Src[i];
		cufftComplex b = d_Dst[i];

		mulAndScale(a, b, c);
		d_Dst[i] = a;

		i += offset;
	}
};

//we use nearest neighbor interpolation to access FFT coefficients in the kernel
__global__ void __launch_bounds__(MAX_THREADS_CUDA)  modulateAndNormalizeSubsampled_kernel(cufftComplex *d_Dst, cufftComplex *d_Src,int kernelDim_0,int kernelDim_1,int kernelDim_2,int imDim_0,int imDim_1,int imDim_2,long long int datasize,float c)
{

	float r_0 = ((float)kernelDim_0) / ((float)imDim_0); //ratio between image size and kernel size to calculate access
	float r_1 = ((float)kernelDim_1) / ((float)imDim_1);
	float r_2 = ((float)kernelDim_2) / ((float)imDim_2);

    long long int i = (long long int)blockDim.x * (long long int)blockIdx.x + (long long int)threadIdx.x;
	long long int offset = (long long int)blockDim.x * (long long int)gridDim.x;
	int k_0,k_1,k_2;
	int aux;
	float auxExp, auxSin,auxCos;
    while(i < datasize)
	{
		//for each dimension we need to access k_i*r_i  i=0, 1, 2
		aux = 1 + imDim_2/2;
		k_2 = i % aux;
		aux = (i - k_2) / aux;
		k_1 = aux % imDim_1;
		k_0 = (aux - k_1) / imDim_1;

		cufftComplex b = d_Dst[i];

		//apply shift in fourier domain since we did not apply fftshift to kernel (so we could use the trick of assuming the kernel is padded with zeros and then just subsample FFT)
		/* This is how we would do it in Matlab (linear phase change)
		auxExp = k_0 * r_0;
		auxExp += k_1 * r_1;
		auxExp += k_2 * r_2;
		auxExp *= PI_1;
		auxSin = sin(auxExp);
		auxCos = cos(auxExp);
		auxExp = b.x * auxCos - b.y * auxSin;

		b.y = b.x * auxSin + b.y * auxCos;
		b.x = auxExp;
		*/

		//add the ratio to each dimension and apply nearest neighbor interpolation
		//k_2 = min((int)(r_2*(float)k_2 + 0.5f),kernelDim_2-1);//the very end points need to be interpolated as "ceiling" instead of round or we can get oout of bounds access
		//k_1 = min((int)(r_1*(float)k_1 + 0.5f),kernelDim_1-1);
		//k_0 = min((int)(r_0*(float)k_0 + 0.5f),kernelDim_0-1);
		k_2 = ((int)(r_2*(float)k_2 + 0.5f)) % kernelDim_2;//the very end points need to be interpolated as "ceiling" instead of round or we can get oout of bounds access
		k_1 = ((int)(r_1*(float)k_1 + 0.5f)) % kernelDim_1;
		k_0 = ((int)(r_0*(float)k_0 + 0.5f)) % kernelDim_0;
		//calculate new coordinate relative to kernel size
		aux = 1 + kernelDim_2/2;
		cufftComplex a = d_Src[k_2 + aux *(k_1 + kernelDim_1 * k_0)];
		
		if( (k_0 + k_1 + k_2) % 2 == 1 )//after much debugging it seems the phase shift is 0 or Pi (nothing in between). In Matlab is a nice linear change as programmed above
		{
			a.x = -a.x;
			a.y = -a.y;
		}
		mulAndScale(a, b, c);

		//__syncthreads();//this actually slows down the code by a lot (0.1 sec for 512x512x512)
		d_Dst[i] = a;

		i += offset;
	}
};

//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
__global__ void __launch_bounds__(MAX_THREADS_CUDA) fftShiftKernel(imageType* kernelCUDA,imageType* kernelPaddedCUDA,int kernelDim_0,int kernelDim_1,int kernelDim_2,int imDim_0,int imDim_1,int imDim_2)
{
	int kernelSize = kernelDim_0 * kernelDim_1 * kernelDim_2;

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid<kernelSize)
	{
		//find coordinates
		long long int x,y,z,aux;
		z = tid % kernelDim_2;
		aux = (tid - z)/kernelDim_2;
		y = aux % kernelDim_1;
		x = (aux - y)/kernelDim_1;

		//center coordinates
		x -= kernelDim_0/2;
		y -= kernelDim_1/2;
		z -= kernelDim_2/2;

		//circular shift if necessary
		if(x<0) x += imDim_0;
		if(y<0) y += imDim_1;
		if(z<0) z += imDim_2;

		//calculate position in padded kernel
		aux = z + imDim_2 * (y + imDim_1 * x);

		//copy value
		kernelPaddedCUDA[aux] = kernelCUDA[tid];//for the most part it should be a coalescent access in oth places
	}
}

//=====================================================================
//-------------to debug elements--------------------------------------
void writeOutCUDAfft(char* filename,imageType* fftCUDA,int* fftCUDAdims)
{
	int fftSize = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		fftSize *= fftCUDAdims[ii];
	}

	//copy FFT from CUDA
	imageType* fftHOST = new imageType[2*fftSize];//complex format
	HANDLE_ERROR(cudaMemcpy(fftHOST,fftCUDA,2*sizeof(imageType)*fftSize,cudaMemcpyDeviceToHost));

	//calculate module
	/*
	int count = 0;
	for(int ii=0;ii<fftSize;ii++)
	{
		fftHOST[ii] = sqrt(fftHOST[count]*fftHOST[count] + fftHOST[count+1]*fftHOST[count+1]);
		count += 2;
	}
	*/

	FILE* fid = fopen(filename,"wb");
	if(fid == NULL)
	{
		printf("ERROR: at writeOutCUDAfft opening file %s\n",filename);
		exit(2);
	}else{
		printf("DEBUGGING: Writing FFT (real part first,imaginary second)  from CUDA of dimensions %d x %d x %d in file %s\n",fftCUDAdims[2],fftCUDAdims[1],fftCUDAdims[0],filename);
	}
	//fwrite(fftHOST,sizeof(imageType),fftSize,fid);
	for(int ii=0;ii<2*fftSize;ii+=2)
		fwrite(&(fftHOST[ii]),sizeof(imageType),1,fid);
	for(int ii=1;ii<2*fftSize;ii+=2)
		fwrite(&(fftHOST[ii]),sizeof(imageType),1,fid);


	fclose(fid);
	delete[] fftHOST;
}


//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
imageType* convolution3DfftCUDA(imageType* im,int* imDim,imageType* kernel,int devCUDA)
{
	imageType* convResult = NULL;
	imageType* imCUDA = NULL;
	imageType* kernelCUDA = NULL;


	cufftHandle fftPlanFwd, fftPlanInv;

	
	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	long long int imSize = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		imSize *= (long long int) (imDim[ii]);
	}

	long long int imSizeFFT = imSize+(long long int)(2*imDim[0]*imDim[1]); //size of the R2C transform in cuFFTComplex

	//allocate memory for output result
	convResult = new imageType[imSize];

	//allocat ememory in GPU
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDA), imSizeFFT*sizeof(imageType) ) );//a little bit larger to allow in-place FFT
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelCUDA), imSizeFFT*sizeof(imageType) ) );


	//TODO: pad image to a power of 2 size in all dimensions (use whatever  boundary conditions you want to apply)
	//TODO: pad kernel to image size
	//TODO: pad kernel and image to xy(z/2 + 1) for in-place transform
	//NOTE: in the example for 2D convolution using FFT in the Nvidia SDK they do the padding in the GPU, but in might be pushing the memory in the GPU for large images.

	//printf("Copying memory (kernel and image) to GPU\n");
	HANDLE_ERROR( cudaMemcpy( kernelCUDA, kernel, imSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( imCUDA, im, imSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );
	
	//printf("Creating R2C & C2R FFT plans for size %i x %i x %i\n",imDim[0],imDim[1],imDim[2]);
	cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL;

	//transforming convolution kernel; TODO: if I do multiple convolutions with the same kernel I could reuse the results at teh expense of using out-of place memory (and then teh layout of the data is different!!!! so imCUDAfft should also be out of place)
	//NOTE: from CUFFT manual: If idata and odata are the same, this method does an in-place transform.
	//NOTE: from CUFFT manual: inplace output data xy(z/2 + 1) with fcomplex. Therefore, in order to perform an in-place FFT, the user has to pad the input array in the last dimension to Nn2 + 1 complex elements interleaved. Note that the real-to-complex transform is implicitly forward.
	cufftExecR2C(fftPlanFwd, imCUDA, (cufftComplex *)imCUDA);HANDLE_ERROR_KERNEL;
	//transforming image
	cufftExecR2C(fftPlanFwd, kernelCUDA, (cufftComplex *)kernelCUDA);HANDLE_ERROR_KERNEL;
	

	//multiply image and kernel in fourier space (and normalize)
	//NOTE: from CUFFT manual: CUFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set followed by an inverse FFT on the resulting set yields data that is equal to the input scaled by the number of elements.
	int numThreads=std::min((long long int)MAX_THREADS_CUDA,imSizeFFT/2);//we are using complex number
	int numBlocks=std::min((long long int)MAX_BLOCKS_CUDA,(long long int)(imSizeFFT/2+(long long int)(numThreads-1))/((long long int)numThreads));
	modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)imCUDA, (cufftComplex *)kernelCUDA, imSizeFFT/2,1.0f/(float)(imSize));//last parameter is the size of the FFT

	//inverse FFT 
	cufftExecC2R(fftPlanInv, (cufftComplex *)imCUDA, imCUDA);HANDLE_ERROR_KERNEL;

	//copy result to host
	HANDLE_ERROR(cudaMemcpy(convResult,imCUDA,sizeof(imageType)*imSize,cudaMemcpyDeviceToHost));

	//release memory
	( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
    ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( imCUDA));
	HANDLE_ERROR( cudaFree( kernelCUDA));

	return convResult;
}

//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 
imageType* convolution3DfftCUDA(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA)
{
	imageType* convResult = NULL;
	imageType* imCUDA = NULL;
	imageType* kernelCUDA = NULL;
	imageType* kernelPaddedCUDA = NULL;


	cufftHandle fftPlanFwd, fftPlanInv;

	
	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	long long int imSize = 1;
	long long int kernelSize = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		imSize *= (long long int) (imDim[ii]);
		kernelSize *= (long long int) (kernelDim[ii]);
	}

	long long int imSizeFFT = imSize+(long long int)(2*imDim[0]*imDim[1]); //size of the R2C transform in cuFFTComplex

	//allocate memory for output result
	convResult = new imageType[imSize];

	//allocat ememory in GPU
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDA), imSizeFFT*sizeof(imageType) ) );//a little bit larger to allow in-place FFT
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelCUDA), (kernelSize)*sizeof(imageType) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelPaddedCUDA), imSizeFFT*sizeof(imageType) ) );


	//TODO: pad image to a power of 2 size in all dimensions (use whatever  boundary conditions you want to apply)
	//TODO: pad kernel to image size
	//TODO: pad kernel and image to xy(z/2 + 1) for in-place transform
	//NOTE: in the example for 2D convolution using FFT in the Nvidia SDK they do the padding in the GPU, but in might be pushing the memory in the GPU for large images.

	//printf("Copying memory (kernel and image) to GPU\n");
	HANDLE_ERROR( cudaMemcpy( kernelCUDA, kernel, kernelSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( imCUDA, im, imSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );

	//apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT
	HANDLE_ERROR( cudaMemset( kernelPaddedCUDA, 0, imSizeFFT*sizeof(imageType) ));
	int numThreads=std::min((long long int)MAX_THREADS_CUDA,kernelSize);
	int numBlocks=std::min((long long int)MAX_BLOCKS_CUDA,(long long int)(kernelSize+(long long int)(numThreads-1))/((long long int)numThreads));
	fftShiftKernel<<<numBlocks,numThreads>>>(kernelCUDA,kernelPaddedCUDA,kernelDim[0],kernelDim[1],kernelDim[2],imDim[0],imDim[1],imDim[2]);HANDLE_ERROR_KERNEL;

	
	//make sure GPU finishes before we launch two different streams
	HANDLE_ERROR(cudaDeviceSynchronize());	

	//printf("Creating R2C & C2R FFT plans for size %i x %i x %i\n",imDim[0],imDim[1],imDim[2]);
	cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL;

	//transforming convolution kernel; TODO: if I do multiple convolutions with the same kernel I could reuse the results at teh expense of using out-of place memory (and then teh layout of the data is different!!!! so imCUDAfft should also be out of place)
	//NOTE: from CUFFT manual: If idata and odata are the same, this method does an in-place transform.
	//NOTE: from CUFFT manual: inplace output data xy(z/2 + 1) with fcomplex. Therefore, in order to perform an in-place FFT, the user has to pad the input array in the last dimension to Nn2 + 1 complex elements interleaved. Note that the real-to-complex transform is implicitly forward.
	cufftExecR2C(fftPlanFwd, imCUDA, (cufftComplex *)imCUDA);HANDLE_ERROR_KERNEL;
	//transforming image
	cufftExecR2C(fftPlanFwd, kernelPaddedCUDA, (cufftComplex *)kernelPaddedCUDA);HANDLE_ERROR_KERNEL;
	

	//multiply image and kernel in fourier space (and normalize)
	//NOTE: from CUFFT manual: CUFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set followed by an inverse FFT on the resulting set yields data that is equal to the input scaled by the number of elements.
	numThreads=std::min((long long int)MAX_THREADS_CUDA,imSizeFFT/2);//we are using complex numbers
	numBlocks=std::min((long long int)MAX_BLOCKS_CUDA,(long long int)(imSizeFFT/2+(long long int)(numThreads-1))/((long long int)numThreads));
	modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)imCUDA, (cufftComplex *)kernelPaddedCUDA, imSizeFFT/2,1.0f/(float)(imSize));//last parameter is the size of the FFT

	//inverse FFT 
	cufftExecC2R(fftPlanInv, (cufftComplex *)imCUDA, imCUDA);HANDLE_ERROR_KERNEL;

	//copy result to host
	HANDLE_ERROR(cudaMemcpy(convResult,imCUDA,sizeof(imageType)*imSize,cudaMemcpyDeviceToHost));

	//release memory
	( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
    ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( imCUDA));
	HANDLE_ERROR( cudaFree( kernelCUDA));
	HANDLE_ERROR( cudaFree( kernelPaddedCUDA));

	return convResult;
}

//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 
void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA)
{
	imageType* imCUDA = NULL;
	imageType* kernelCUDA = NULL;
	imageType* kernelPaddedCUDA = NULL;


	cufftHandle fftPlanFwd, fftPlanInv;

	
	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	long long int imSize = 1;
	long long int kernelSize = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		imSize *= (long long int) (imDim[ii]);
		kernelSize *= (long long int) (kernelDim[ii]);
	}

	long long int imSizeFFT = imSize+(long long int)(2*imDim[0]*imDim[1]); //size of the R2C transform in cuFFTComplex

	//allocat ememory in GPU
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDA), imSizeFFT*sizeof(imageType) ) );//a little bit larger to allow in-place FFT
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelCUDA), (kernelSize)*sizeof(imageType) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelPaddedCUDA), imSizeFFT*sizeof(imageType) ) );

	//TODO: pad image to a power of 2 size in all dimensions (use whatever  boundary conditions you want to apply)
	//TODO: pad kernel to image size
	//TODO: pad kernel and image to xy(z/2 + 1) for in-place transform
	//NOTE: in the example for 2D convolution using FFT in the Nvidia SDK they do the padding in the GPU, but in might be pushing the memory in the GPU for large images.

	//printf("Copying memory (kernel and image) to GPU\n");
	HANDLE_ERROR( cudaMemcpy( kernelCUDA, kernel, kernelSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( imCUDA, im, imSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );


	//apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT
	HANDLE_ERROR( cudaMemset( kernelPaddedCUDA, 0, imSizeFFT*sizeof(imageType) ));
	int numThreads=std::min((long long int)MAX_THREADS_CUDA,kernelSize);
	int numBlocks=std::min((long long int)MAX_BLOCKS_CUDA,(long long int)(kernelSize+(long long int)(numThreads-1))/((long long int)numThreads));
	fftShiftKernel<<<numBlocks,numThreads>>>(kernelCUDA,kernelPaddedCUDA,kernelDim[0],kernelDim[1],kernelDim[2],imDim[0],imDim[1],imDim[2]);HANDLE_ERROR_KERNEL;

	
	//make sure GPU finishes 
	HANDLE_ERROR(cudaDeviceSynchronize());	

	//printf("Creating R2C & C2R FFT plans for size %i x %i x %i\n",imDim[0],imDim[1],imDim[2]);
	cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	//cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);HANDLE_ERROR_KERNEL;//we wait to conserve memory
	//cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL;

	//transforming convolution kernel; TODO: if I do multiple convolutions with the same kernel I could reuse the results at teh expense of using out-of place memory (and then teh layout of the data is different!!!! so imCUDAfft should also be out of place)
	//NOTE: from CUFFT manual: If idata and odata are the same, this method does an in-place transform.
	//NOTE: from CUFFT manual: inplace output data xy(z/2 + 1) with fcomplex. Therefore, in order to perform an in-place FFT, the user has to pad the input array in the last dimension to Nn2 + 1 complex elements interleaved. Note that the real-to-complex transform is implicitly forward.
	cufftExecR2C(fftPlanFwd, imCUDA, (cufftComplex *)imCUDA);HANDLE_ERROR_KERNEL;
	//transforming image
	cufftExecR2C(fftPlanFwd, kernelPaddedCUDA, (cufftComplex *)kernelPaddedCUDA);HANDLE_ERROR_KERNEL;

	//int fftCUDAdims[dimsImage] ={imDim[0],imDim[1],1+imDim[2]/2};
	//writeOutCUDAfft("E:/temp/fftCUDAgood.bin",kernelPaddedCUDA,fftCUDAdims);

	//multiply image and kernel in fourier space (and normalize)
	//NOTE: from CUFFT manual: CUFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set followed by an inverse FFT on the resulting set yields data that is equal to the input scaled by the number of elements.
	numThreads=std::min((long long int)MAX_THREADS_CUDA,imSizeFFT/2);//we are using complex numbers
	numBlocks=std::min((long long int)MAX_BLOCKS_CUDA,(long long int)(imSizeFFT/2+(long long int)(numThreads-1))/((long long int)numThreads));
	modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)imCUDA, (cufftComplex *)kernelPaddedCUDA, imSizeFFT/2,1.0f/(float)(imSize));HANDLE_ERROR_KERNEL;//last parameter is the size of the FFT


	//we destroy memory first so we can perform larger block convoutions
	( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( kernelCUDA));
	HANDLE_ERROR( cudaFree( kernelPaddedCUDA));

	//inverse FFT 
	cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL;
	cufftExecC2R(fftPlanInv, (cufftComplex *)imCUDA, imCUDA);HANDLE_ERROR_KERNEL;
	

	//copy result to host and overwrite image
	HANDLE_ERROR(cudaMemcpy(im,imCUDA,sizeof(imageType)*imSize,cudaMemcpyDeviceToHost));

	//release memory
	( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
    //( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( imCUDA));
	//HANDLE_ERROR( cudaFree( kernelCUDA));
	//HANDLE_ERROR( cudaFree( kernelPaddedCUDA));

}


//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 
/*
void convolution3DfftCUDAInPlaceSaveMemory(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA)
{
	imageType* imCUDA = NULL;
	imageType* kernelCUDA = NULL;
	//imageType* kernelFFTCUDA = NULL; //I need a duplicate in order to perform FFTshift (I can not do it in place)

	cufftHandle fftPlanFwd, fftPlanInv;

	
	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	long long int imSize = 1;
	long long int kernelSize = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		imSize *= (long long int) (imDim[ii]);
		kernelSize *= (long long int) (kernelDim[ii]);
	}

	long long int imSizeFFT = imSize+(long long int)(2*imDim[0]*imDim[1]); //size of the R2C transform in cuFFTComplex
	long long int kernelSizeFFT = kernelSize + (long long int)(2*kernelDim[0]*kernelDim[1]);

	

	//allocat ememory in GPU
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDA), imSizeFFT*sizeof(imageType) ) );//a little bit larger to allow in-place FFT
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelCUDA), (kernelSizeFFT)*sizeof(imageType) ) );
	//HANDLE_ERROR( cudaMalloc( (void**)&(kernelFFTCUDA), (kernelSizeFFT)*sizeof(imageType) ) );

	//TODO: pad image to a power of 2 size in all dimensions (use whatever  boundary conditions you want to apply)
	//TODO: pad kernel to image size
	//TODO: pad kernel and image to xy(z/2 + 1) for in-place transform
	//NOTE: in the example for 2D convolution using FFT in the Nvidia SDK they do the padding in the GPU, but in might be pushing the memory in the GPU for large images.

	//printf("Copying memory (kernel and image) to GPU\n");
	HANDLE_ERROR( cudaMemcpy( kernelCUDA, kernel, kernelSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( imCUDA, im, imSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );


	//apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT
	//HANDLE_ERROR( cudaMemset( kernelFFTCUDA, 0, kernelSizeFFT*sizeof(imageType) ));
	//int numThreads=std::min((long long int)MAX_THREADS_CUDA,kernelSize);
	//int numBlocks=std::min((long long int)MAX_BLOCKS_CUDA,(long long int)(kernelSize+(long long int)(numThreads-1))/((long long int)numThreads));
	//fftShiftKernel<<<numBlocks,numThreads>>>(kernelCUDA,kernelFFTCUDA,kernelDim[0],kernelDim[1],kernelDim[2],kernelDim[0],kernelDim[1],kernelDim[2]);HANDLE_ERROR_KERNEL;

	
	//make sure GPU finishes 
	HANDLE_ERROR(cudaDeviceSynchronize());	

	//printf("Creating R2C & C2R FFT plans for size %i x %i x %i\n",imDim[0],imDim[1],imDim[2]);
	cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	//transforming convolution kernel; TODO: if I do multiple convolutions with the same kernel I could reuse the results at teh expense of using out-of place memory (and then teh layout of the data is different!!!! so imCUDAfft should also be out of place)
	//NOTE: from CUFFT manual: If idata and odata are the same, this method does an in-place transform.
	//NOTE: from CUFFT manual: inplace output data xy(z/2 + 1) with fcomplex. Therefore, in order to perform an in-place FFT, the user has to pad the input array in the last dimension to Nn2 + 1 complex elements interleaved. Note that the real-to-complex transform is implicitly forward.
	cufftExecR2C(fftPlanFwd, imCUDA, (cufftComplex *)imCUDA);HANDLE_ERROR_KERNEL;


	//transforming kernel
	cufftDestroy(fftPlanFwd); HANDLE_ERROR_KERNEL;
	cufftPlan3d(&fftPlanFwd, kernelDim[0], kernelDim[1], kernelDim[2], CUFFT_R2C);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
	cufftExecR2C(fftPlanFwd, kernelCUDA, (cufftComplex *)kernelCUDA);HANDLE_ERROR_KERNEL;


	//int fftCUDAdims[dimsImage] ={kernelDim[0],kernelDim[1],1+kernelDim[2]/2};
	//writeOutCUDAfft("E:/temp/fftCUDAbad.bin",kernelCUDA,fftCUDAdims);

	//multiply image and kernel in fourier space (and normalize)
	//NOTE: from CUFFT manual: CUFFT performs un-normalized FFTs; that is, performing a forward FFT on an input data set followed by an inverse FFT on the resulting set yields data that is equal to the input scaled by the number of elements.
	int numThreads=std::min((long long int)MAX_THREADS_CUDA,imSizeFFT/2);//we are using complex numbers
	int numBlocks=std::min((long long int)MAX_BLOCKS_CUDA,(long long int)(imSizeFFT/2+(long long int)(numThreads-1))/((long long int)numThreads));
	//we multiply two FFT of different sizes (but one was just padded with zeros)
	modulateAndNormalizeSubsampled_kernel<<<numBlocks,numThreads>>>((cufftComplex *)imCUDA, (cufftComplex *)kernelCUDA, kernelDim[0],kernelDim[1],kernelDim[2],imDim[0],imDim[1],imDim[2], imSizeFFT/2,1.0f/(float)(imSize));HANDLE_ERROR_KERNEL;


	//we destroy memory first so we can perform larger block convoutions
	cufftDestroy(fftPlanFwd) ;HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( kernelCUDA));
	//HANDLE_ERROR( cudaFree( kernelFFTCUDA));

	//inverse FFT 
	cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);HANDLE_ERROR_KERNEL;
	cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL;
	cufftExecC2R(fftPlanInv, (cufftComplex *)imCUDA, imCUDA);HANDLE_ERROR_KERNEL;
	

	//copy result to host and overwrite image
	HANDLE_ERROR(cudaMemcpy(im,imCUDA,sizeof(imageType)*imSize,cudaMemcpyDeviceToHost));

	//release memory
	( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( imCUDA));
	
}
*/
