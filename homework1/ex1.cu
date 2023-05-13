#include "ex1.h"

__device__ void prefixSum(int arr[], int len, int tid, int threads) {
    //TODO
}

__device__ void argmin(int arr[], int len, int tid, int threads) {
    assert(threads == len / 2);
    int halfLen = len / 2;
    bool firstIteration = true;
    int prevHalfLength = 0;
    while (halfLen > 0) {
        if(tid < halfLen){
            if(arr[tid] == arr[tid + halfLen]){ //a corenr case
                int lhsIdx = tid;
                int rhdIdx = tid + halfLen;
                int lhsOriginalIdx = firstIteration ? lhsIdx : arr[prevHalfLength + lhsIdx];
                int rhsOriginalIdx = firstIteration ? rhdIdx : arr[prevHalfLength + rhdIdx];
                arr[tid + halfLen] = lhsOriginalIdx < rhsOriginalIdx ? lhsOriginalIdx : rhsOriginalIdx;
            }
            else{ //the common case
                bool isLhsSmaller = (arr[tid] < arr[tid + halfLen]);
                int idxOfSmaller = isLhsSmaller * tid + (!isLhsSmaller) * (tid + halfLen);
                int smallerValue = arr[idxOfSmaller];
                int origIdxOfSmaller = firstIteration * idxOfSmaller + (!firstIteration) * arr[prevHalfLength + idxOfSmaller];
                arr[tid] = smallerValue;
                arr[tid + halfLen] = origIdxOfSmaller;
            }
        }
        __syncthreads();
        firstIteration = false;
        prevHalfLength = halfLen;
        halfLen /= 2;
    }
}

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS]){
    //TODO
}

__device__ void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    //TODO    
}

__global__
void process_image_kernel(uchar *targets, uchar *refrences, uchar *results) {
    // TODO
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context* task_serial_init()
{
    auto context = new task_serial_context;

    //TODO: allocate GPU memory for a single input image and a single output image

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial

}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input and output images.
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all input images and all output images

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out

}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}


/********************************************************
**  the following waappers are needed for unit testing.
********************************************************/

__global__ void argminWrapper(int arr[], int size){
    argmin(arr, size, threadIdx.x, blockDim.x);
}

__global__ void colorHistWrapper(uchar img[][CHANNELS], int histograms[][LEVELS]){
    __shared__ int histogramsSahred[CHANNELS][LEVELS];

    int tid = threadIdx.x;;
    int threads = blockDim.x;

    colorHist(img, histogramsSahred);

    __syncthreads();

    for(int i = tid; i < CHANNELS * LEVELS; i+=threads){
        ((int*)histograms)[i] = ((int*)histogramsSahred)[i];
    }
    
}

__global__ void prefixSumWrapper(int arr[], int size){
    __shared__ int arrSahred[LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for(int i=tid; i<size; i+=threads){
        arrSahred[i] = arr[i];
    }

    __syncthreads();

    prefixSum(arrSahred, size, threadIdx.x, blockDim.x);

    for(int i=tid; i<size; i+=threads){
        arr[i] = arrSahred[i];
    }

    __syncthreads();
}

__global__ void performMappingWrapper(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    performMapping(maps, targetImg, resultImg);
}
