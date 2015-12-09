// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>


// Simple little unit for timing using the gettimeofday() call.
// By Ingemar 2009

#include <stdlib.h>
#include <sys/time.h>

static struct timeval timeStart;
static char hasStart = 0;

const int64_t N = 1024;
const int64_t blocksize = 16;
const int64_t gridsize = N / blocksize;

int GetMilliseconds()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) / 1000 + (tv.tv_sec - timeStart.tv_sec)*1000;
}

int GetMicroseconds()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) + (tv.tv_sec - timeStart.tv_sec)*1000000;
}

double GetSeconds()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (double)(tv.tv_usec - timeStart.tv_usec) / 1000000.0 + (double)(tv.tv_sec - timeStart.tv_sec);
}

// If you want to start from right now.
void ResetMilli()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	hasStart = 1;
	timeStart = tv;
}

// If you want to start from a specific time.
void SetMilli(int seconds, int microseconds)
{
	hasStart = 1;
	timeStart.tv_sec = seconds;
	timeStart.tv_usec = microseconds;
}


void add_matrix(float *a, float *b, float *c, int64_t N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}



__global__ 
void simple(float *c, float *a, float *b) 
{
	unsigned int blockId = gridDim.y * blockIdx.x + blockIdx.y;
	unsigned int index = blockId * blockDim.x * blockDim.y + blockDim.y * threadIdx.x + threadIdx.y;
	c[index] = a[index] + b[index];
}

int main()
{
	// Get GPU properties
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	printf("Name: %s \n", properties.name);
	printf("Warp size: %i \n", properties.warpSize);
	printf("Max Threads per Block: %i \n", properties.maxThreadsPerBlock);
	printf("Max Threads, x: %i, y: %i, z: %i \n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
	printf("Max Grid Size, x: %i, y: %i, z: %i \n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);

	// Init vairables
	float *c = new float[N*N];
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *cd, *ad, *bd;
	const int64_t size = N*N*sizeof(float);
	
	float timeGPU;
	double timeCPU;

	cudaEvent_t startEvent;
	cudaEventCreate(&startEvent);
	cudaEvent_t endEvent;
	cudaEventCreate(&endEvent);
	

	// Fill memory
	for (int64_t j = 0; j < N; j++) {
		for (int64_t i = 0; i < N; i++) {
			a[i + j*N] = (float)i;
			b[i + j*N] = (float)j / 10000.0f;
		}
	}
	
	// Perform CPU matrix add
	timeCPU = GetSeconds(); // Start CPU timer
	add_matrix(a, b, c, N);
	timeCPU = GetSeconds(); // End CPU timer
	
	printf("Time taken (CPU): %f\n", timeCPU);

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
	cudaEventElapsedTime(&timeGPU, startEvent, endEvent);


	/*
	for (int j = N-8; j < N; j++) {
		for (int i = N-8; i < N; i++) {
			printf("%0.4f ", c[i + j*N]);
		}
		printf("\n");
	}
	*/

	printf("Time taken (GPU): %f\n", timeGPU / 1000.f);
	delete[] c;
	delete[] a;
	delete[] b;
	printf("done\n");
	return EXIT_SUCCESS;
}
