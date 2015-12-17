// Reduction lab, find maximum

#include <stdio.h>
#include "milli.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

bool isPow4(unsigned int x) {
	if (x & (x - 1)) return false; // Check power of 2
	return (x & 0x55555555) != 0; // Check that only an odd bit is set. 
}

bool isPow2(unsigned int x) {
	if (x == 0) return false;
	return (x & (x - 1)) == 0;
}

unsigned int makePow2(unsigned int x) {
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x++;
	return x;
}

unsigned int makePow4(unsigned int x) {
	x = makePow2(x);
	return isPow4(x) ? x : x * 2;
}
/*
__device__ int max(int x, int y) {
	return x > y ? x : y;
}
*/

#define THREADS 1024

__device__ void warpReduce(volatile int* sData, int thid) {
	sData[thid] = max(sData[thid], sData[thid + 32]);
	sData[thid] = max(sData[thid], sData[thid + 16]);
	sData[thid] = max(sData[thid], sData[thid + 8]);
	sData[thid] = max(sData[thid], sData[thid + 4]);
	sData[thid] = max(sData[thid], sData[thid + 2]);
	sData[thid] = max(sData[thid], sData[thid + 1]);
}

__global__ void find_max(int *indata, int *outdata) {
	__shared__ int sData[THREADS];

	int i = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
	int thid = threadIdx.x;

	sData[thid] = max(indata[i], indata[i + blockDim.x]);
	__syncthreads();
	
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (thid < s) {
			sData[thid] = max(sData[thid], sData[thid + s]);
		}
		__syncthreads();
	}
	
	if (thid < 32) {
		warpReduce(sData, thid);
	}
	
	if (thid == 0) outdata[blockIdx.x] = sData[0];

}

cudaEvent_t startEvent;
cudaEvent_t endEvent;



void launch_cuda_kernel(int *data, int N) {
	// Handle your CUDA kernel launches in this function

	int size = sizeof(int)* N;

	int* devdata0;
	int* devdata1;

	cudaMalloc((void**)&devdata0, size);
	cudaMemcpy(devdata0, data, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&devdata1, size);
	cudaMemset(devdata1, 0, size);

	// Dummy launch
	int numThreads = THREADS;
	int elemPerThread = 2;
	int numBlocks = N / numThreads / elemPerThread;
	dim3 dimBlock(numBlocks, 1);
	dim3 dimGrid(numThreads, 1);

	cudaEventRecord(startEvent, 0);
	find_max << < dimBlock, dimGrid >> >(devdata0, devdata1);

	std::swap(devdata0, devdata1);

	find_max << < 1, dimGrid >> > (devdata0, devdata1);
	cudaEventRecord(endEvent, 0);

	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(devdata0);
	cudaFree(devdata1);
}

// CPU max finder (sequential)
void find_max_cpu(int *data, int N) {
	int i, m;

	m = data[0];
	for (i = 0; i<N; i++) // Loop over data
	{
		if (data[i] > m)
		
			m = data[i];
	}
	data[0] = m;
}

#define SIZE THREADS * THREADS * 2
//#define SIZE 
// Dummy data in comments below for testing
int data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

int main() {
	Timer timer;
	float timeGPU = 0.0f;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&endEvent);

	// Get GPU properties
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	printf("Name: %s \n", properties.name);
	printf("Warp size: %i \n", properties.warpSize);
	printf("Max Threads per Block: %i \n", properties.maxThreadsPerBlock);
	printf("Max Threads, x: %i, y: %i, z: %i \n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
	printf("Max Grid Size, x: %i, y: %i, z: %i \n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);

	// Generate 2 copies of random data
	srand(time(NULL));
	timer.StartTimer();
	for (long i = 0; i < SIZE; i++) {
		data[i] = rand();
		data2[i] = data[i];
	}
	// The GPU will not easily beat the CPU here!
	// Reduction needs optimizing or it will be slow.
	timer.EndTimer();
	find_max_cpu(data, SIZE);
	timer.EndTimer();
	printf("\nCPU time %f sec\n", timer.GetSeconds());
	printf("CPU time %f sec (incl setup)\n", timer.GetTotalSeconds());


	timer.EndTimer();
	launch_cuda_kernel(data2, SIZE);
	timer.EndTimer();
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&timeGPU, startEvent, endEvent);

	printf("\nGPU time %f sec\n", timeGPU / 1000.0f);
	printf("GPU time %f sec (incl setup)\n", timer.GetSeconds());

	// Print result
	printf("\n");
	printf("Max should be close to: %i\n", RAND_MAX);
	printf("CPU found max %i\n", data[0]);
	printf("GPU found max %i\n", data2[0]);
	printf("Press enter to quit...");
	getchar();

}
