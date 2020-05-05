#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" void allocateMemory(int **arr, int arraySize)
{
     cudaMallocManaged(arr, ( (arraySize* sizeof(int))));
}

extern "C" void callCudaFree(int* local)
{
	cudaFree(local);
}

//extern void callMPI(int* local,int* arr,int arrSize,int mpi_size,int x_rank);


extern "C" void cudaInit( int myrank)
{
	int cE;
    int cudaDeviceCount = 1;

    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n",
        cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
        myrank, (myrank % cudaDeviceCount), cE);
        exit(-1);
    }
}

__global__ void mergeKernel(int j, int mpi_size, int mpi_rank, int *arr, int arrSize, int sizeCompare,int* prev_local, int* next_local)
{	
	//nt *prev_local = NULL;
    //int *next_local = NULL;

	bool sameVal = false;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_idx = i + arrSize / mpi_size * mpi_rank;
    int x = global_idx ^ j;
    int x_rank = x / (arrSize / mpi_size);
    if ( global_idx >= x ) {
        if ( mpi_rank == x_rank ) {
            if(sameVal == false)
            {
                sameVal = true;
            }
        } 
        else {
            if ( prev_local == NULL ) {
                //prev_local = calloc(arrSize / mpi_size, sizeof(int));
                //allocateMemory(&prev_local,arrSize/mpi_size);
                prev_local = arr + arrSize / mpi_size * x_rank;
                //callMPI(prev_local,arr,arrSize,mpi_size,x_rank);
            }

            if ( (sizeCompare & x) == 0 && arr[i] < prev_local[i] ) {
                arr[i] = prev_local[i];
            }
            if ( (sizeCompare & x) != 0 && arr[i] > prev_local[i] ) {
                arr[i] = prev_local[i];
            }
        }
    }
    else { 
        if ( x_rank == mpi_rank ) {
            int y = x - arrSize / mpi_size * mpi_rank;
            if ( (global_idx & sizeCompare) == 0 && arr[i] > arr[y] ) {
                int temp = arr[i];
                arr[i] = arr[y];
                arr[y] = temp;
            }
            if ( (global_idx & sizeCompare) != 0 && arr[i] < arr[y] ) {
                int temp = arr[i];
                arr[i] = arr[y];
                arr[y] = temp;
            }
        }
        else {
            if ( next_local == NULL ) {
                //next_local = calloc(arrSize / mpi_size, sizeof(int));
                //allocateMemory(&next_local,arrSize/mpi_size);
                next_local = arr + arrSize / mpi_size * x_rank;
                //callMPI(next_local,arr,arrSize,mpi_size,x_rank);
            }

            if ( (global_idx & sizeCompare) == 0 && arr[i] > next_local[i] ) {
                arr[i] = next_local[i];
            }
            if ( (global_idx & sizeCompare) != 0 && arr[i] < next_local[i] ) {
                arr[i] = next_local[i];
            }
        }
    }
}

extern "C" void mergeKernelLaunch(int blockSize,int threadsCount,int j, int mpi_size, int mpi_rank, int *arr, int arrSize, int sizeCompare,int* prev_local, int* next_local)
{
	mergeKernel<<<blockSize,threadsCount>>>(j, mpi_size, mpi_rank, arr, arrSize, sizeCompare, prev_local, next_local);
}