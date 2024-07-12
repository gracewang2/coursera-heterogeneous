// Tiled Image Convolution
// pitch is set to width in this MP
// 3-channel (RGB) image
#include <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width   5
#define Mask_radius  Mask_width/2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH  (O_TILE_WIDTH + Mask_width - 1)
#define CHANNELS     3 // RGB

//@@ INSERT CODE HERE
// kernel code
__global__ void convolution(float * input, float * output, float * mask, int width, int height) {
    __shared__ float shared_tile[CHANNELS][BLOCK_WIDTH][BLOCK_WIDTH];
    int channel = blockIdx.z;

    int row_o = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
    int col_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int index_o = (row_o * width + col_o) * CHANNELS + channel; // linearized index in output

    int row_i = row_o - Mask_radius;
    int col_i = col_o - Mask_radius;
    int index_i = (row_i * width + col_i) * CHANNELS + channel; // linearized index in input

    //@@ Load a tile of the input into shared memory
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) shared_tile[channel][threadIdx.y][threadIdx.x] = input[index_i];
    else shared_tile[channel][threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    //@@ Calculate output value from shared memory - not all threads participate in calculating output
    float output_value = 0;
    if (row_o < height && col_o < width && threadIdx.y < O_TILE_WIDTH && threadIdx.x < O_TILE_WIDTH) {
        for (int i = 0; i < Mask_width; i++) {
            for (int j = 0; j < Mask_width; j++) {
                output_value += mask[i * Mask_width + j] * shared_tile[channel][threadIdx.y + i][threadIdx.x + j];
            }
        }
        __syncthreads();
        output[index_o] = output_value;
    }
}


// host code
int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid((imageWidth - 1)/O_TILE_WIDTH + 1, (imageHeight - 1)/O_TILE_WIDTH + 1, CHANNELS); // # blocks in grid
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1); // # threads per block

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    convolution<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, imageWidth, imageHeight);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
