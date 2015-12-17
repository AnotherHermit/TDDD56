/*
 * Image filter in OpenCL
 */


#define KERNELSIZE 4
#define FILTERSIZE (2*KERNELSIZE+1)
#define DIVVAL (FILTERSIZE*FILTERSIZE)
#define LOCALWIDTH 16
#define LOAD_PIXEL(i, j, I, J) \
  lImage[0 + 1*(i + (j) * LOCALWIDTH)] = image[0+3 * ((I) + (J) * 512)];\
  lImage[1 + 2*(i + (j) * LOCALWIDTH)] = image[1+3 * ((I) + (J) * 512)];\
  lImage[2 + 0*(i + (j) * LOCALWIDTH)] = image[2+3 * ((I) + (J) * 512)];

struct vec2
{
  int x;
  int y;
};


__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m)
{ 
  struct vec2 gtid = {get_global_id(0), get_global_id(1)};
  struct vec2 tid = {get_local_id(0), get_local_id(1)};
  
  int k, l;
  unsigned int sumx = 0, sumy = 0, sumz = 0;
   
  __local unsigned char lImage[256 * 3];

  if (gtid.x - 4 >= 0 && gtid.y - 4 >= 0)
    LOAD_PIXEL(tid.x + 0, tid.y + 0, gtid.x - 4, gtid.y - 4)
  if (gtid.x + 4 < 512 && gtid.y - 4 >= 0)
    LOAD_PIXEL(tid.x + 8, tid.y + 0, gtid.x + 4, gtid.y - 4)
  if (gtid.x - 4 >= 0 && gtid.y + 4 < 512)
    LOAD_PIXEL(tid.x + 0, tid.y + 8, gtid.x - 4, gtid.y + 4)
  if (gtid.x + 4 < 512 && gtid.y + 4 < 512)
    LOAD_PIXEL(tid.x + 8, tid.y + 8, gtid.x + 4, gtid.y + 4)


  if (gtid.x < KERNELSIZE || gtid.x + KERNELSIZE >= 512 ||
      gtid.y < KERNELSIZE || gtid.y + KERNELSIZE >= 512)
  {
    out[0+3 * (gtid.x + gtid.y * 512)] = image[0+3 * (gtid.x + gtid.y * 512)];
    out[1+3 * (gtid.x + gtid.y * 512)] = image[1+3 * (gtid.x + gtid.y * 512)];
    out[2+3 * (gtid.x + gtid.y * 512)] = image[2+3 * (gtid.x + gtid.y * 512)];
    return;
  }
  	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(k=0;k<FILTERSIZE;k++) 
	{
		for(l=0;l<FILTERSIZE;l++)	
		{		
			sumx += lImage[((tid.x+k)+LOCALWIDTH*(tid.y+l))*3+0];
			sumy += lImage[((tid.x+k)+LOCALWIDTH*(tid.y+l))*3+1];
		  sumz += lImage[((tid.x+k)+LOCALWIDTH*(tid.y+l))*3+2];
		}
	}
	
	if (gtid.x + gtid.y < 10) sumx = sumy = sumz = DIVVAL*255;

	out[0+3 * (gtid.x + gtid.y * 512)] = sumx / DIVVAL;
	out[1+3 * (gtid.x + gtid.y * 512)] = sumy / DIVVAL;
	out[2+3 * (gtid.x + gtid.y * 512)] = sumz / DIVVAL;
}
