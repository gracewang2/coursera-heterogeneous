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

//@@ INSERT CODE HERE

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
    const char * inputImageFile;

    //@@ INSERT MORE CODE HERE

    args = wbArg_read(argc, argv); /* parse the input arguments */
    
    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ INSERT CODE HERE

    wbSolution(args, outputImage);

    //@@ INSERT CODE HERE

    return 0;
}

