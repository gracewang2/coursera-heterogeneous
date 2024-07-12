// Histogram Equalization
// I'm still confused about the objective and specifics of this lab, so I'll work on this later.
/*
Essentially, histogram equalization is a technique used in image processing to improve the contrast of an image. In the end, we want
the histogram of pixel intensity values to become roughly uniform. This can be done through the following steps:
1. Calculate the histogram
2. Compute the Cumulative Distribution Function (CDF)
3. Normalize the CDF - scale the CDF so that the maximum value is equal to the maximum intensity value of the image
4. Map the original intensity values to new intensity values (from normalized CDF)
*/

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512

// 1. converting image array from float to unsigned char
// input = deviceInputImageData, output = deviceRGBData, len = imageWidth * imageHeight * imageChannels
__global__ void floatToUChar(float * input, unsigned char * output, int len) {
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < len; i += stride) output[i] = (unsigned char) (255 * input[i]);
}

// 2. converting image array from RGB to grayscale
// PROVIDED: (r, g, b) -> 0.21 * r + 0.71 * g + 0.07 * b
// input = deviceRGBData, output = deviceGrayData
__global__ void RGBToGrayscale(unsigned char * input, unsigned char * output, int imageWidth, int imageHeight) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // output index of pixel (roughly 1/3 of input index of pixel)
    if (index < imageWidth * imageHeight) {
        int red = input[3 * index];
        int green = input[3 * index + 1];
        int blue = input[3 * index + 2];
        output[index] = (unsigned char) (0.21 * red + 0.71 * green + 0.07 * blue);
    }
}

// 3. calculating histogram from grayscale data
// int atomicAdd(int * address, int val)
// input = deviceGrayscaleData, histogram = deviceHistogram
__global__ void calculateHistogram(unsigned char * input, unsigned int * histogram, int imageWidth, int imageHeight) {
    // threads in the same block access a shared histogram (privatization technique)
    __shared__ unsigned int private_histogram[HISTOGRAM_LENGTH];

    // initialize the bin counters in private copies of histogram
    if (threadIdx.x < HISTOGRAM_LENGTH) private_histogram[threadIdx.x] = 0;
    __syncthreads();

    // build private histogram
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in grid
    while (index < imageWidth * imageHeight) {
        atomicAdd(&private_histogram[input[index]], 1);
        index += stride;
    }

    // wait for all other threads in the block to finish
    __syncthreads();

    // accumulate results into output
    if (threadIdx.x < HISTOGRAM_LENGTH) atomicAdd(&histogram[threadIdx.x], private_histogram[threadIdx.x]);
}

// 4. calculate histogram CDF in single-block scan kernel, similar to mp5 (parallel scan)
// histogram = deviceHistogram, CDF = deviceCDF
__global__ void calculateHistogramCDF(unsigned int * histogram, float * CDF, int imageWidth, int imageHeight) {
    __shared__ unsigned int XY[HISTOGRAM_LENGTH];

    // load elements from histogram into XY
    if (threadIdx.x < HISTOGRAM_LENGTH) XY[threadIdx.x] = histogram[threadIdx.x];
    else XY[threadIdx.x] = 0;
    if (threadIdx.x + blockDim.x < HISTOGRAM_LENGTH) XY[threadIdx.x + blockDim.x] = histogram[threadIdx.x + blockDim.x];
    else XY[threadIdx.x + blockDim.x] = 0;
    __syncthreads();
    
    // Reduction Phase (Down-sweep)
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x) XY[index] += XY[index - stride];
        __syncthreads();
    }

    // Post-Reduction Reverse Phase (Up-sweep)
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < 2 * blockDim.x) XY[index + stride] += XY[index];
        __syncthreads();
    }

    // write to CDF and scale by probability
    float probability = 1.0 / (imageWidth * imageHeight);
    if (threadIdx.x < HISTOGRAM_LENGTH) CDF[threadIdx.x] = XY[threadIdx.x] * probability;
    if (threadIdx.x + blockDim.x < HISTOGRAM_LENGTH) CDF[threadIdx.x + blockDim.x] = XY[threadIdx.x + blockDim.x] * probability;
}

// 5. using the histogram CDF, equalize the image and scale the pixel channels
// corrected image value computed as (CDF[init_val] - CDF[0]) / (1 - CDF[0]) * 255
// CDF = deviceCDF, image = deviceRGBData, len = imageWidth * imageHeight * imageChannels
__global__ void equalizeImage(float * CDF, unsigned char * image, int len) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float min_cdf = CDF[0]; // this is so that we don't have to access it repeatedly
    for (int i = index; i < len; i += stride) {
        unsigned char init_val = image[i];
        int corrected_val = (int) ((CDF[init_val] - min_cdf) / (1.0 - min_cdf) * 255);
        if (corrected_val > 255) corrected_val = 255;
        if (corrected_val < 0) corrected_val = 0;
        image[i] = corrected_val;
    }
}

// 6. convert unsigned char image array back into float
__global__ void uCharToFloat(unsigned char * input, float * output, int len) {
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < len; i += stride) output[i] = (float) (input[i] / 255.0);
}

// host code
int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * deviceInputImageData; // newly added
    float * deviceOutputImageData; // newly added
    const char * inputImageFile;
    unsigned char * deviceRGBData; // newly added
    unsigned char * deviceGrayData; // newly added
    unsigned int * deviceHistogram; // newly added
    float * deviceCDF; // newly added

    args = wbArg_read(argc, argv); // parse the input arguments
    
    inputImageFile = wbArg_getInputFile(args, 0);

    //@@ Importing Data/Creating Memory on Host
    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ GPU Memory Allocation
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceRGBData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMalloc((void **) &deviceGrayData, imageWidth * imageHeight * sizeof(unsigned char));
    cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
    cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    //@@ Histogram Initialization
    cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));

    //@@ Copying Data to GPU
    wbTime_start(GPU, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying data to the GPU");

    //@@ GPU Computation
    wbTime_start(GPU, "Doing the computation on the GPU");
    int imageSize = imageWidth * imageHeight * imageChannels;
    int imagePixels = imageWidth * imageHeight;

    // 1. converting image array from float to unsigned char
    dim3 DimGrid_1((imageSize - 1) / BLOCK_SIZE + 1, 1, 1); // # blocks in grid
    dim3 DimBlock_1(BLOCK_SIZE, 1, 1); // # threads per block
    floatToUChar<<<DimGrid_1, DimBlock_1>>>(deviceInputImageData, deviceRGBData, imageSize);

    // 2. converting image array from RGB to grayscale
    dim3 DimGrid_2((imagePixels - 1) / BLOCK_SIZE + 1, 1, 1); // # blocks in grid
    dim3 DimBlock_2(BLOCK_SIZE, 1, 1); // # threads per block
    RGBToGrayscale<<<DimGrid_2, DimBlock_2>>>(deviceRGBData, deviceGrayData, imageWidth, imageHeight);

    // 3. calculating histogram from grayscale data
    dim3 DimGrid_3((imagePixels - 1) / BLOCK_SIZE + 1, 1, 1); // # blocks in grid
    dim3 DimBlock_3(BLOCK_SIZE, 1, 1); // # threads per block
    calculateHistogram<<<DimGrid_3, DimBlock_3>>>(deviceGrayData, deviceHistogram, imageWidth, imageHeight);

    // 4. calculate histogram CDF in single-block scan kernel
    dim3 DimGrid_4(1, 1, 1); // # blocks in grid
    dim3 DimBlock_4(HISTOGRAM_LENGTH, 1, 1); // # threads per block
    calculateHistogramCDF<<<DimGrid_4, DimBlock_4>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);

    // 5. using the histogram CDF, equalize the image and scale the pixel channels
    dim3 DimGrid_5((imageSize - 1) / BLOCK_SIZE + 1, 1, 1); // # blocks in grid
    dim3 DimBlock_5(BLOCK_SIZE, 1, 1); // # threads per block
    equalizeImage<<<DimGrid_5, DimBlock_5>>>(deviceCDF, deviceRGBData, imageSize);

    // 6. convert unsigned char image array back into float
    dim3 DimGrid_6((imageSize - 1) / BLOCK_SIZE + 1, 1, 1); // # blocks in grid
    dim3 DimBlock_6(BLOCK_SIZE, 1, 1); // # threads per block
    uCharToFloat<<<DimGrid_6, DimBlock_6>>>(deviceRGBData, deviceOutputImageData, imageSize);

    cudaDeviceSynchronize();

    wbTime_stop(GPU, "Doing the computation on the GPU");

    //@@ Copying Data from GPU
    wbTime_start(GPU, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(GPU, "Copying data from the GPU");

    //@@ Output the Solution
    wbSolution(args, outputImage);

    //@@ Free GPU Memory
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceRGBData);
    cudaFree(deviceGrayData);
    cudaFree(deviceHistogram);
    cudaFree(deviceCDF);
    wbTime_stop(GPU, "Freeing GPU Memory");

    //@@ Free Host Memory
    wbImage_delete(inputImage);
    wbImage_delete(outputImage);

    return 0;
}



