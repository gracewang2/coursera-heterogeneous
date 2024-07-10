// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void accumulate(float * blockTotalSums, float * output, int len) {
    //@@ blockTotalSums refers to the total sum of each block
    //@@ Used to calculate the comprehensive prefix sums
    int index = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    for (int j = 0; j < blockIdx.x; j++) {
        if (index < len) output[index] += blockTotalSums[j];
        if (index + blockDim.x < len) output[index + blockDim.x] += blockTotalSums[j];
        __syncthreads();
    }
}

__global__ void scan(float * input, float * output, float * blockTotalSums, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    // # threads/block is half of # elements/block
    // if each block has 8 elements, we would have threads @ x1, x3, x5, x7
    __shared__ float XY[2 * BLOCK_SIZE];

    //@@ Initialization of XY
    int index = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    if (index < len) XY[threadIdx.x] = input[index]; // loading from first half
    else XY[threadIdx.x] = 0;

    if (index + blockDim.x < len) XY[threadIdx.x + blockDim.x] = input[index + blockDim.x]; // loading from second half
    else XY[threadIdx.x + blockDim.x] = 0;

    __syncthreads(); // Don't forget to __syncthreads() here!

    //@@ Reduction Phase (Down-sweep)
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int i = (threadIdx.x + 1) * stride * 2 - 1;
        if (i < 2 * BLOCK_SIZE) XY[i] += XY[i - stride];
        __syncthreads();
    }

    //@@ Post-Reduction Reverse Phase (Up-sweep)
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int i = (threadIdx.x + 1) * stride * 2 - 1;
        if (i + stride < 2 * BLOCK_SIZE) XY[i + stride] += XY[i];
    }
    __syncthreads();
    if (index < len) output[index] = XY[threadIdx.x];
    if (index + blockDim.x < len) output[index + blockDim.x] = XY[threadIdx.x + blockDim.x];

    //@@ Fill blockTotalSums
    // for example, if 2 * BLOCK_SIZE = 8, we want the element at index 7
    // because the sum at that element is the sum of the entire block
    blockTotalSums[blockIdx.x] = output[2 * blockDim.x * (blockIdx.x + 1) - 1];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float * blockTotalSums; // the total sums of each individual block - to be used as an intermediate result for comprehensive prefix sums
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    int dimGrid = (numElements - 1) / (BLOCK_SIZE * 2) + 1; // # blocks in grid, there are 2 * BLOCK_SIZE elements/block
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&blockTotalSums, dimGrid*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbCheck(cudaMemset(blockTotalSums, 0, dimGrid*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(dimGrid, 1, 1); // # blocks in grid
    dim3 DimBlock(BLOCK_SIZE, 1, 1); // # threads per block
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the device
    scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, blockTotalSums, numElements);
    cudaDeviceSynchronize();
    accumulate<<<DimGrid, DimBlock>>>(blockTotalSums, deviceOutput, numElements);
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(blockTotalSums);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


