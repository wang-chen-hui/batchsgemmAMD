#include <stdio.h>
#include <string.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;



#include <cstdlib>
#include <cstdio>
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_profile.h>
using std::vector;


int main( void ) {
    hipDeviceProp_t  prop;
 
    //int count;
    //HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    int count = 1;
    for (int i=0; i< count; i++) 
    {
        ( hipGetDeviceProperties( &prop, i ) );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap:  " );
        /*
	if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
	
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
	*/ 
        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        //printf( "Max mem pitch:  %ld\n", prop.memPitch );
        //printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
 
        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}
