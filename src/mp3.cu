#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

const int TILE_WIDTH = 16;

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float Cvalue = 0;

  // Loop over the A and B tiles required to compute the C element
  for (int c = 0; c < (numAColumns - 1)/TILE_WIDTH + 1; c++) {
    // Collaborative loading of A and B tiles into shared memory
    if (row < numARows && c * TILE_WIDTH + threadIdx.x < numAColumns) {
      ds_A[threadIdx.y][threadIdx.x] = A[row * numAColumns + c * TILE_WIDTH + threadIdx.x];
    }
    else ds_A[threadIdx.y][threadIdx.x] = 0;

    if (c * TILE_WIDTH + threadIdx.y < numBRows && col < numBColumns) {
      ds_B[threadIdx.y][threadIdx.x] = B[(c * TILE_WIDTH + threadIdx.y) * numBColumns + col];
    } 
    else ds_B[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; i++) {
      Cvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < numCRows && col < numCColumns) C[row * numCColumns + col] = Cvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC =
      ( float * )malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int size_a = numARows * numAColumns * sizeof(float);
  int size_b = numBRows * numBColumns * sizeof(float);
  int size_c = numCRows * numCColumns * sizeof(float);
  cudaMalloc((void **) &deviceA, size_a);
  cudaMalloc((void **) &deviceB, size_b);
  cudaMalloc((void **) &deviceC, size_c);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, size_b, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numCColumns - 1)/TILE_WIDTH + 1, (numCRows - 1)/TILE_WIDTH + 1, 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
                                              numARows, numAColumns,
                                              numBRows, numBColumns,
                                              numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, size_c, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
