
// This is not really C++-code but pretty plain C code, but we compile it
// as C++ so we can integrate with CUDA seamlessly.

// If you plan on submitting your solution for the Parallel Sorting Contest,
// please keep the split into main file and kernel file, so we can easily
// insert other data.

#include <stdio.h>
#include "bitonic_kernel.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ static inline void exchange(int *i, int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

// No, this is not GPU code yet but just a copy of the CPU code, but this
// is where I want to see your GPU code!

__global__ void bitonic_block(int* data, int k, int j)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int ixj=i^j; // Calculate indexing
  
  if ((ixj)>i)
  {
    if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
    if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
  }
}


float bitonic_gpu(int *data, int N)
{
  int j,k;
  
  float timeGPU;
  
  int* devdata;
  int size = sizeof(int) * N;

  dim3 dimBlock(N / 1024, 1);
  dim3 dimGrid(1024, 1);

  cudaEvent_t startEvent;
  cudaEvent_t endEvent;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&endEvent);

	cudaMalloc((void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice);
  
	cudaEventRecord(startEvent, 0);
  for (k=2;k<=N;k=2*k) // Outer loop, double size for each step
  {
    for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
    {
      bitonic_block<<< dimBlock, dimGrid >>>(devdata, k, j);
    }
  }
  cudaEventRecord(endEvent, 0);
  
  cudaEventSynchronize(endEvent);
  cudaEventElapsedTime(&timeGPU, startEvent, endEvent);
  
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, size, cudaMemcpyDeviceToHost);
	cudaFree(devdata);
	

	
	return timeGPU;
}
