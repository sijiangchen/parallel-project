#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h> 

void merge( int a[], int low, int arrSize, int dir ) { 
    if ( arrSize > 1 ) { 
        int halfSize = arrSize / 2; 
        for ( int i = low; i < low + halfSize; i++ ) { 
            if ( dir == ( a[i] > a[i + halfSize] ) ) {
                int temp = a[i];
                a[i] = a[i + halfSize];
                a[i + halfSize] = temp;
            }
        }
        merge( a, low, halfSize, dir ); 
        merge( a, low + halfSize, halfSize, dir ); 
    } 
} 

void sort( int a[], int low, int arrSize, int dir ) { 
    if ( arrSize > 1 ) { 
        int halfSize = arrSize / 2; 
        sort( a, low, halfSize, 1 ); 
        sort( a, low + halfSize, halfSize, 0 );  
        merge( a, low, arrSize, dir ); 
    } 
} 

int main( int argc, char *argv[] ) { 
    int i; 
    int arrSize = atoi( argv[1] );
    int *arr = calloc( arrSize, sizeof( int ) );
    for ( i = 0; i < arrSize; i++ ) {
        arr[i] = rand();
    } 
    // Time start
    sort( arr, 0, arrSize, 1 ); 
    // Time end

    for ( i = 1; i < arrSize; i++ ) {
        if ( arr[i - 1] > arr[i] ) {
            fprintf(stderr, "%d, %d\n", arr[i - 1], arr[i]);
        }
    }
    
    
    printf( "Final Array: \n" ); 
    for ( int i = 0; i < arrSize; i++ ) { 
        printf( "%d\n", arr[i] );
    }
    
    return 0; 
} 
