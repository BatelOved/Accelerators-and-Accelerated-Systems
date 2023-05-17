#include "ex1.h"

__device__ void prefixSum(int arr[], int len, int tid, int threads)
{
    assert(threads >= len);
    int increment;
    
    // Kogge-Stone Algorithm
    for (int stride = 1; stride < len; stride *= 2) {
        if (tid < len && tid >= stride) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        
        if (tid < len && tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__device__ void argmin(int arr[], int len, int tid, int threads)
{
    assert(threads == len / 2);
    int halfLen = len / 2;
    bool firstIteration = true;
    int prevHalfLength = 0;
    while (halfLen > 0)
    {
        if (tid < halfLen)
        {
            if (arr[tid] == arr[tid + halfLen])
            { // a corner case
                int lhsIdx = tid;
                int rhdIdx = tid + halfLen;
                int lhsOriginalIdx = firstIteration ? lhsIdx : arr[prevHalfLength + lhsIdx];
                int rhsOriginalIdx = firstIteration ? rhdIdx : arr[prevHalfLength + rhdIdx];
                arr[tid + halfLen] = lhsOriginalIdx < rhsOriginalIdx ? lhsOriginalIdx : rhsOriginalIdx;
            }
            else
            { // the common case
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

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS])
{
    int tid = threadIdx.x;

    if (tid < LEVELS)
    {
        for (int j = 0; j < CHANNELS; j++)
        {
            histograms[j][tid] = 0;
        }
    }

    __syncthreads();

    for (int i = tid; i < SIZE * SIZE; i += blockDim.x)
    {
        uchar *rgbPixel = img[i];
        for (int j = 0; j < CHANNELS; j++)
        {
            int *channelHist = histograms[j];
            atomicAdd(&channelHist[rgbPixel[j]], 1);
        }
    }

    __syncthreads();
}

__device__ void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS])
{
    int tid = threadIdx.x;

    for (int i = tid; i < SIZE * SIZE; i += blockDim.x)
    {
        uchar *inRgbPixel = targetImg[i];
        uchar *outRgbPixel = resultImg[i];

        for (int j = 0; j < CHANNELS; j++)
        {
            uchar *mapChannel = maps[j];
            outRgbPixel[j] = mapChannel[inRgbPixel[j]];
        }
    }
    __syncthreads();
}

__device__ void calculateMap(uchar map[][LEVELS], int target[][LEVELS], int reference[][LEVELS])
{
    int tid = threadIdx.x;
    __shared__ int diff_row[LEVELS];

    for (int i = 0; i < CHANNELS; i++) {
        for(int i_tar = 0; i_tar < LEVELS; i_tar++) {
            for(int i_ref = tid; i_ref < LEVELS; i_ref += blockDim.x) {
                diff_row[tid] = abs(reference[i][tid] - target[i][i_tar]);
            }
            __syncthreads();

            argmin(diff_row, LEVELS, tid, LEVELS/2);
            __syncthreads();

            if (tid == 0) {
                map[i][i_tar] = diff_row[1];
            }
            __syncthreads();
        }
    }
}

__global__ void process_image_kernel(uchar *targets, uchar *refrences, uchar *results)
{
    assert(blockDim.x % 2 == 0);

    __shared__ int histograms_tar[CHANNELS][LEVELS];
    __shared__ int histograms_ref[CHANNELS][LEVELS];
    __shared__ uchar map[CHANNELS][LEVELS];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    uchar *img_target = &targets[bid * SIZE * SIZE * CHANNELS];
    uchar *img_refrence = &refrences[bid * SIZE * SIZE * CHANNELS];
    uchar *img_out = &results[bid * SIZE * SIZE * CHANNELS];


    /*Calculate Histograms*/
    colorHist((uchar(*)[CHANNELS])img_target, histograms_tar);
    colorHist((uchar(*)[CHANNELS])img_refrence, histograms_ref);
    __syncthreads();

    /*Calculate Reference and target Histogram-Prefix sum*/
    for (int k = 0; k < CHANNELS; k++) {
        prefixSum(histograms_tar[k], LEVELS, tid, blockDim.x);
        prefixSum(histograms_ref[k], LEVELS, tid, blockDim.x);
    }
    __syncthreads();

    calculateMap(map, histograms_tar, histograms_ref);
    __syncthreads();

    performMapping(map, (uchar(*)[CHANNELS])img_target, (uchar(*)[CHANNELS])img_out);
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context
{
    uchar *inputBuffer_tar;
    uchar *inputBuffer_ref;
    uchar *outputBuffer;
};

/* Allocate GPU memory for a single input image and a single output image.
 *
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    cudaMalloc(&context->inputBuffer_tar, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->inputBuffer_ref, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->outputBuffer, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_target = &context->inputBuffer_tar[i * SIZE * SIZE * CHANNELS];
        uchar *img_refrence = &context->inputBuffer_ref[i * SIZE * SIZE * CHANNELS];
        uchar *img_out = &context->outputBuffer[i * SIZE * SIZE * CHANNELS];

        cudaMemcpy(img_target, &images_target[i * SIZE * SIZE * CHANNELS], SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(img_refrence, &images_refrence[i * SIZE * SIZE * CHANNELS], SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
        
        process_image_kernel<<<1, 1024>>>(img_target, img_refrence, img_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&images_result[i * SIZE * SIZE * CHANNELS], img_out, SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyDeviceToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    cudaFree(context->inputBuffer_ref);
    cudaFree(context->inputBuffer_tar);
    cudaFree(context->outputBuffer);
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context
{
    uchar *inputBuffer_tar;
    uchar *inputBuffer_ref;
    uchar *outputBuffer;
};

/* Allocate GPU memory for all the input and output images.
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    cudaMalloc(&context->inputBuffer_tar, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->inputBuffer_ref, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));
    cudaMalloc(&context->outputBuffer, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    uchar *img_target = context->inputBuffer_tar;
    uchar *img_refrence = context->inputBuffer_ref;
    uchar *img_out = context->outputBuffer;

    cudaMemcpy(img_target, images_target, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(img_refrence, images_refrence, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyHostToDevice);
    process_image_kernel<<<N_IMAGES, 1024>>>(img_target, img_refrence, img_out);
    cudaDeviceSynchronize();

    cudaMemcpy(images_result, img_out, N_IMAGES * SIZE * SIZE * CHANNELS * sizeof(uchar), cudaMemcpyDeviceToHost);
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    cudaFree(context->inputBuffer_ref);
    cudaFree(context->inputBuffer_tar);
    cudaFree(context->outputBuffer);
    free(context);
}

/********************************************************
**  the following waappers are needed for unit testing.
********************************************************/

__global__ void argminWrapper(int arr[], int size)
{
    argmin(arr, size, threadIdx.x, blockDim.x);
}

__global__ void colorHistWrapper(uchar img[][CHANNELS], int histograms[][LEVELS])
{
    __shared__ int histogramsSahred[CHANNELS][LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    colorHist(img, histogramsSahred);

    __syncthreads();

    for (int i = tid; i < CHANNELS * LEVELS; i += threads)
    {
        ((int *)histograms)[i] = ((int *)histogramsSahred)[i];
    }
}

__global__ void prefixSumWrapper(int arr[], int size)
{
    __shared__ int arrSahred[LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for (int i = tid; i < size; i += threads)
    {
        arrSahred[i] = arr[i];
    }

    __syncthreads();

    prefixSum(arrSahred, size, threadIdx.x, blockDim.x);

    for (int i = tid; i < size; i += threads)
    {
        arr[i] = arrSahred[i];
    }

    __syncthreads();
}

__global__ void performMappingWrapper(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS])
{
    performMapping(maps, targetImg, resultImg);
}