// Refer to https://zhuanlan.zhihu.com/p/545641318

#include <stdio.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <device_launch_parameters.h>

// Define a global macro to check for errors
#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int args, char ** argv) 
{
    int device_count = 0;
    // Get the number of GPUs on the current machine
    CHECK(cudaGetDeviceCount(&device_count));
    printf("Device count %d\n", device_count);

    // Iterate through all GPUs and get their properties
    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        // Get the properties of the current GPU
        CHECK(cudaGetDeviceProperties(&prop, i));
        // Set the GPU to be used by its index
        cudaSetDevice(i);
        // avail: available GPU memory, total: total GPU memory
        size_t avail;
        size_t total;
        cudaMemGetInfo( &avail, &total );
        
        printf("Device name %s\n", prop.name);
        // Total memory size
        printf("Amount of global memory: %g GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        // Total memory and available memory
        printf("Amount of total memory: %g GB avail memory: %g GB\n", total / (1024.0 * 1024.0 * 1024.0), avail / (1024.0 * 1024.0 * 1024.0));
        /**< Global memory bus width in bits */
        printf("Global memory bus width in bits:   %d bit\n", prop.memoryBusWidth);
        // Compute capability: identifies the core architecture of the device, the features and instructions supported by the GPU hardware, sometimes referred to as "SM version"
        printf("Compute capability:     %d.%d\n", prop.major, prop.minor);
        // Constant memory size
        printf("Amount of constant memory:      %g KB\n", prop.totalConstMem / 1024.0);
        // Maximum grid size
        printf("Maximum grid size:  %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        // Maximum block size
        printf("maximum block size:     %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        // Number of SMs
        printf("Number of SMs:      %d\n", prop.multiProcessorCount);
        printf("Number of warpSize:      %d\n", prop.warpSize);
        /**< Size of L2 cache in bytes */
        printf("L2 Cache size:                             %d KB\n", prop.l2CacheSize / 1024);
        /**< Device's maximum l2 persisting lines capacity setting in bytes */
        printf("maximum l2 persisting lines capacity       %d B\n", prop.persistingL2CacheMaxSize);
        /**< Device supports caching globals in L1 */
        printf("Device supports caching globals in L1(Y/N) %d\n", prop.globalL1CacheSupported);
        /**<  Device supports caching locals in L1 */
        printf("Device supports caching locals in L1(Y/N)  %d\n", prop.localL1CacheSupported);
        // Maximum amount of shared memory per block
        printf("Maximum amount of shared memory per block: %g KB\n", prop.sharedMemPerBlock / 1024.0);
        // Maximum amount of shared memory per SM
        printf("Maximum amount of shared memory per SM:    %g KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
        // Maximum number of registers per block
        printf("Maximum number of registers per block:     %d K\n", prop.regsPerBlock / 1024);
        // Maximum number of registers per SM
        printf("Maximum number of registers per SM:        %d K\n", prop.regsPerMultiprocessor / 1024);
        // Maximum number of threads per block
        printf("Maximum number of threads per block:       %d\n", prop.maxThreadsPerBlock);
        // Maximum number of threads per SM
        printf("Maximum number of threads per SM:          %d\n", prop.maxThreadsPerMultiProcessor);
    }
}
