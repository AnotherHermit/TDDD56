
// This is not really C++-code but pretty plain C code, but we compile it
// as C++ so we can integrate with CUDA seamlessly.

#include "bitonic_kernel.h"
#include <stdio.h>
#include "milli.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SIZE 1024 * 1
#define MAXPRINTSIZE 32
int data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

static void exchange(int *i, int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

void bitonic_cpu(int *data, int N)
{
  int i,j,k;
  for (k=2;k<=N;k=2*k) // Outer loop, double size for each step
  {
    for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
    {
      for (i=0;i<N;i++) // Loop over data
      {
        int ixj=i^j; // Calculate indexing!
        if ((ixj)>i)
        {
          if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
          if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
        }
      }
    }
  }
}


int main()
{
  Timer timer;
  float timeGPU = 0.0f;

	srand(time(NULL));
	timer.StartTimer();
	for (long i = 0; i < SIZE; i++) {
		data[i] = rand();
		data2[i] = data[i];
	}
	
  timer.EndTimer();
  bitonic_cpu(data, SIZE);
  timer.EndTimer();
  printf("\nCPU: %f sec\n", timer.GetSeconds());
  printf("CPU: %f sec (incl setup)\n", timer.GetTotalSeconds());

  timer.EndTimer();
  timeGPU = bitonic_gpu(data2, SIZE);
  timer.EndTimer();

  printf("\nGPU: %f sec\n", timeGPU / 1000.0f);
  printf("GPU: %f sec (incl setup)\n", timer.GetSeconds());

  for (int i=0;i<SIZE;i++)
    if (data[i] != data2[i])
    {
      printf("Error at %d ", i);
      return(1);
    }

  // Print result
  if (SIZE <= MAXPRINTSIZE)
    for (int i=0;i<SIZE;i++)
      printf("%d ", data[i]);
  printf("\nYour sorting looks correct!\n");
}
