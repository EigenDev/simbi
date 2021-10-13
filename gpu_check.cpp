/***
 * Check for available GPUs on system
 * if there are any. Return boolean value at
 * end. Adapted from https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
*/

#include <stdio.h>
#include "hip/hip_runtime.h"

int main()
{
    int nDevices;

    hipGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        hipDeviceProp prop;
        hipGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }

    /* don't just return the number of gpus, because other runtime cuda
    errors can also yield non-zero return values */
    if (nDevices > 0)
        return 0; /* success */
    else
        return 1; /* failure */
}