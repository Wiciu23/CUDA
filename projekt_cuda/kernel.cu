
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cstdio>

__global__ void addTen(float* d, int wielkosc){

	int ThreadsInBlock = blockDim.x * blockDim.y* blockDim.z;

	int PositionThreadInBlock = threadIdx.x + threadIdx.y* blockDim.x + threadIdx.z * blockDim.x*blockDim.y;

	int PositionBlockInGrid = blockIdx.x + blockIdx.y*gridDim.x;

	int index = PositionBlockInGrid * ThreadsInBlock + PositionThreadInBlock;

	if(index < wielkosc){
		d[index] += 10;
	}
}

int main()
{

  curandGenerator_t generator;
  curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(generator,time(0));

   const int wielkosc = 123456;
   int rozmiar = sizeof(float)*wielkosc;
   float tablica[wielkosc];
   float *d;
   cudaMalloc(&d,rozmiar);
   curandGenerateUniform(generator,d,wielkosc);

   dim3 block(8,8,8);
   dim3 grid(16,16);

   addTen<<<grid,block>>>(d,wielkosc);

   cudaMemcpy(tablica,d,rozmiar,cudaMemcpyDeviceToHost);

   cudaFree(d);

   for (int i = 0; i < wielkosc; i++)
   {
	   printf("\n %f", tablica[i]);
   }

}