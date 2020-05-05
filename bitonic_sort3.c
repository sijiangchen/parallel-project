#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h> 
#include <mpi.h>

#define GetTimeBase MPI_Wtime

extern void allocateMemory(int** arr, int arrSize);

extern void mergeKernelLaunch(int blockSize,int threadsCount,int j, int mpi_size, int mpi_rank, int *arr, int arrSize, int sizeCompare, int* prev_local,int* next_local);

extern void callCudaFree(int* prev_local);

typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
unsigned int tbl, tbu0, tbu1;
do {
__asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
__asm__ __volatile__ ("mftb %0" : "=r"(tbl));
__asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
} while (tbu0 != tbu1);
return ((((unsigned long long)tbu0) << 32) | tbl);
}


// Merges elements of array for one iteration in bitonic sort
void merge (int j, int mpi_size, int mpi_rank, int *arr, int arrSize, int sizeCompare) {
    MPI_Request req_next, req_prev, send_req_next, send_req_prev;
    MPI_Status stat;
    bool sameVal = false;
    int i, global_idx;
    int *prev_local = NULL;
    int *next_local = NULL;
    
    int threadsCount = 32;
    
    
     for ( i = 0; i < arrSize / mpi_size; i++ ) 
        {
            global_idx = i + arrSize / mpi_size * mpi_rank;
            int x = global_idx ^ j;
            int x_rank = x / (arrSize / mpi_size);
            if(global_idx>=x)
            {
                if(mpi_rank != x_rank && prev_local == NULL)
                {
                    allocateMemory(&prev_local,arrSize/mpi_size);
                    MPI_Irecv(prev_local, arrSize / mpi_size, MPI_INT, x_rank, 0, MPI_COMM_WORLD, &req_prev);
                    MPI_Isend(arr, arrSize / mpi_size, MPI_INT, x_rank, 0, MPI_COMM_WORLD, &send_req_prev);
                    MPI_Wait(&req_prev, &stat);
                    MPI_Wait(&send_req_prev, &stat);
                }   
            }
            else
            {
                if(mpi_rank != x_rank && next_local == NULL)
                {
                    allocateMemory(&next_local,arrSize/mpi_size);
                    MPI_Irecv(next_local, arrSize / mpi_size, MPI_INT, x_rank, 0, MPI_COMM_WORLD, &req_next);
                    MPI_Isend(arr, arrSize / mpi_size, MPI_INT, x_rank, 0, MPI_COMM_WORLD, &send_req_next);
                    MPI_Wait(&req_next, &stat);
                    MPI_Wait(&send_req_next, &stat);
                }
            }
            if(prev_local !=NULL && next_local != NULL)
                break;
        }
    
        mergeKernelLaunch(arrSize/mpi_size/threadsCount,threadsCount,j, mpi_size, mpi_rank, arr, arrSize, sizeCompare,prev_local,next_local);
        
    if ( next_local != NULL ) {
            callCudaFree(next_local);
           
    }
    if ( prev_local != NULL ) {
            callCudaFree(prev_local);
           
    }

}


extern void cudaInit( int myrank);





int main (int argc, char *argv[])
{
    setvbuf( stdout, NULL, _IONBF, 0 );
    
    int i, j, sizeCompare, arrSize;
    int *arr = NULL;
    int mpi_size = -1;
    int mpi_rank = -1;
    MPI_File fh;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Status status;
    
    if ( argc == 2 ) {
        arrSize = atoi(argv[1]);
    }
    else {
        perror("Incorrect number of arguments. Please specify array size.\n");
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }
    
    cudaInit(mpi_rank);  
    

    if ( mpi_rank == 0 ) {
        allocateMemory(&arr,arrSize);
        for ( i = 0; i < arrSize; i++ ) {
            arr[i] = rand();
        }
    }
      
   // MPI_Request req1, req2;
        
    unsigned long long start = getticks(); 
    double start_time = MPI_Wtime();
    sizeCompare = arrSize / mpi_size;
    

    if ( mpi_rank != 0 ) {
        allocateMemory(&arr,arrSize);
        MPI_Scatter(arr, sizeCompare, MPI_INT, arr, sizeCompare, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatter(arr, sizeCompare, MPI_INT, MPI_IN_PLACE, sizeCompare, MPI_INT, 0, MPI_COMM_WORLD);
    }
    

    //mpi io operation- write
    unsigned long long write_start;
    double write_s;
           if(mpi_rank==0) {
                write_start=getticks();
               write_s=MPI_Wtime();
               
           }
    
     int nints = arrSize/ mpi_size;
     MPI_File_open( MPI_COMM_WORLD, "bitonic", MPI_MODE_RDWR | MPI_MODE_CREATE , MPI_INFO_NULL, &fh );
       
       //write the input to the file
       MPI_Offset offset = mpi_rank *nints*sizeof(int) ;
      
       //each process will has its file view which starts with offset
        MPI_File_set_view(fh, offset, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
       
        MPI_Offset  seek_offset = 0;
        int whence = MPI_SEEK_SET;
        MPI_File_seek(fh, seek_offset, whence);
        int write_buf[nints];
        
       for(int i=0;i<nints;++i){
           write_buf[i]=arr[i];
       }
    
        
        MPI_File_write_all(fh, write_buf, nints, MPI_INT, &status);
        MPI_File_close(&fh);
    
    
       unsigned long long write_end;
        double write_e;
         if(mpi_rank==0) {
              write_end=getticks();
             write_e=MPI_Wtime();
             
             // double write_time=write_end-write_start;
              printf("mpi write wtime is: %f seconds\n",write_e-write_s);
             
             long double write_time = ((long double)(write_end-write_start))/512000000;
             printf("mpi write time is: %Lf seconds\n",write_time);
          }
    

    for ( sizeCompare = 2; sizeCompare <= arrSize; sizeCompare *= 2 ) { 
        for ( j = sizeCompare >> 1; j > 0; j = j >> 1 ) { 
            merge(j, mpi_size, mpi_rank, arr, arrSize, sizeCompare);
            //MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    
    if ( mpi_rank != 0 ) {
        MPI_Gather(arr, arrSize / mpi_size, MPI_INT, arr, arrSize / mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Gather(MPI_IN_PLACE, arrSize / mpi_size, MPI_INT, arr, arrSize / mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    
    if ( mpi_rank != 0 ) {
        callCudaFree(arr);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
   // double timeEnd = MPI_Wtime();
    
    unsigned long long finish;
    double end_time;
     if(mpi_rank==0){
        end_time= MPI_Wtime();
        finish = getticks();
         
    }
    
    //mpi io
        MPI_File_open( MPI_COMM_WORLD, "bitonic", MPI_MODE_RDWR | MPI_MODE_CREATE , MPI_INFO_NULL, &fh );
       
         MPI_Offset offset2 = mpi_rank *nints*sizeof(int) ;
       
        //each process will has its file view which starts with offset
         MPI_File_set_view(fh, offset2, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
        MPI_Offset seek_offset2 = 0;
        int whence2 = MPI_SEEK_SET;
        //after set fileview, the seek is the context of new fileview
        MPI_File_seek(fh, seek_offset2, whence2);
       
        //read start with seek's position
        int read_buf[nints];
       
      //blocking call
        MPI_File_read_all(fh, read_buf, nints, MPI_INT, &status);
    
       //MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_close(&fh);
       
         unsigned long long read_end;
             double read_e;
         if(mpi_rank==0) {
            read_e=MPI_Wtime();
             read_end = getticks();
             //double read_time=read_end-write_end;
             printf("mpi read wtime is: %f seconds\n",read_e-end_time);
             long double read_time = ((long double)(read_end-finish))/512000000;
            
             printf("mpi read time is: %Lf seconds\n",read_time);
         }
    
    
    //debug
    /*
    for(int i=0;i<nints;++i){
        printf("%d ",read_buf[i]);
    }
    printf("\n");
    */
    //printing stuff, unrealted to algo
    if ( mpi_rank == 0 ) {
        /*
        for ( i = 1; i < arrSize; i++ ) {
            if ( arr[i - 1] > arr[i] ) {
                fprintf(stderr, "%d, %d\n", arr[i - 1], arr[i]);
            }
        }*/
       // double execution_time = ((double)(timeEnd - timeStart));
      //  unsigned long long finish = getticks();
       // printf("Algorithm finished in %f seconds.\n", execution_time);
      //  fflush(stdout);
        /*
        bool printFlag = false;
        if(printFlag)
            for( i = 0; i < arrSize; i++)
                printf("%d,", arr[i]);*/
        callCudaFree(arr);
       // printf("\n\n");
        //printf("Ticks: %llu - %llu = %llu\n",finish,start,finish-start);
        double endtime=(end_time-start_time+write_s-write_e);
        printf("wtime end in %f\n",endtime);
        long double tickTime = ((long double)(finish-start+write_start-write_end))/512000000;
        printf("Algorithm finished in %Lf seconds.\n",tickTime);

    }



    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

