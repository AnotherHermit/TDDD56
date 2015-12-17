/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 4

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m)
{ 
  unsigned int i = get_global_id(1) % 512;
  unsigned int j = get_global_id(0) % 512;
  int k, l;
  unsigned int sumx, sumy, sumz;

	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
	
	if (j < n && i < m) // If inside image
	{
		if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
		{
		// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
				for(l=-KERNELSIZE;l<=KERNELSIZE;l++)	
				{
					sumx += image[((i+k)*n+(j+l))*3+0];
					sumy += image[((i+k)*n+(j+l))*3+1];
					sumz += image[((i+k)*n+(j+l))*3+2];
				}
			out[(i*n+j)*3+0] = sumx/divby;
			out[(i*n+j)*3+1] = sumy/divby;
			out[(i*n+j)*3+2] = sumz/divby;
		}
		else
		// Edge pixels are not filtered
		{
			out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
			out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
			out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
		}
	}
}
