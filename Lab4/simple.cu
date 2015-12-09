// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

const __int64 N = 1024; //048;
const __int64 blocksize = 16; // 2048;
const __int64 gridsize = N / blocksize;

__global__ 
void simple(float *c, float *a, float *b) 
{
	unsigned int blockId = gridDim.y * blockIdx.x + blockIdx.y;
	unsigned int index = blockId * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
	c[index] = a[index] + b[index];
}

int main()
{
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	printf("Warp size: %i \n", properties.warpSize);
	printf("Max Threads per Block: %i \n", properties.maxThreadsPerBlock);
	printf("Max Threads, x: %i, y: %i, z: %i \n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
	printf("Max Grid Size, x: %i, y: %i, z: %i \n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);


	float *c = new float[N*N];
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *cd, *ad, *bd;
	const __int64 size = N*N*sizeof(float);
	
	float time;

	cudaEvent_t startEvent;
	cudaEventCreate(&startEvent);
	

	cudaEvent_t endEvent;
	cudaEventCreate(&endEvent);
	


	for (__int64 j = 0; j < N; j++) {
		for (__int64 i = 0; i < N; i++) {
			a[i + j*N] = (float)i;
			b[i + j*N] = (float)j / 100000000.0f;
		}
	}
	

	cudaMalloc( (void**)&cd, size );
	cudaMalloc((void**)&ad, size);
	cudaMalloc((void**)&bd, size);
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid( gridsize, gridsize );

	cudaEventRecord(startEvent, 0);
	simple<<<dimGrid, dimBlock>>>(cd, ad, bd);
	cudaEventRecord(endEvent, 0);

	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	cudaFree(ad);
	cudaFree(bd);
	
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&time, startEvent, endEvent);


	
	for (int j = N-8; j < N; j++) {
		for (int i = N-8; i < N; i++) {
			printf("%0.4f ", c[i + j*N]);
		}
		printf("\n");
	}
	

	printf("Time taken: %f\n", time);
	delete[] c;
	delete[] a;
	delete[] b;
	printf("done\n");
	return EXIT_SUCCESS;
}
