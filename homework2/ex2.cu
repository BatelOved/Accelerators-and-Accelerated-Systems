#include "ex2.h"
#include <cuda/atomic>

#define QUEUE_SIZE 16

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

__device__ void process_image(uchar *targets, uchar *refrences, uchar *results)
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

/**********************************************************************************************************************/

/* Job context struct with necessary CPU / GPU pointers to process a single image */
struct job_context {
    typedef enum {AVAILABLE = -1} job_status;

    uchar *target;
    uchar *reference;
    uchar *result;
    int job_id;
};

__global__
void process_image_kernel(uchar *target, uchar *reference, uchar *result) {
    process_image(target, reference, result);
}

class streams_server : public image_processing_server
{
private:
    cudaStream_t streams[STREAM_COUNT];
    job_context  jobs[STREAM_COUNT];

public:
    streams_server()
    {
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&(jobs[i].target), IMG_BYTES));
            CUDA_CHECK(cudaMalloc(&(jobs[i].reference), IMG_BYTES));
            CUDA_CHECK(cudaMalloc(&(jobs[i].result), IMG_BYTES));
            jobs[i].job_id = job_context::AVAILABLE;
        }
    }

    ~streams_server() override
    {
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(jobs[i].target));
            CUDA_CHECK(cudaFree(jobs[i].reference));
            CUDA_CHECK(cudaFree(jobs[i].result));
        }
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        for (int i = 0; i < STREAM_COUNT; i++) {
            if (jobs[i].job_id == job_context::AVAILABLE) {
                jobs[i].job_id = job_id;
                CUDA_CHECK(cudaMemcpyAsync(jobs[i].target, target, IMG_BYTES, cudaMemcpyHostToDevice, streams[i]));
                CUDA_CHECK(cudaMemcpyAsync(jobs[i].reference, reference, IMG_BYTES, cudaMemcpyHostToDevice, streams[i]));
                process_image_kernel<<<1, 1024, 0, streams[i]>>>(jobs[i].target, jobs[i].reference, jobs[i].result);
                CUDA_CHECK(cudaMemcpyAsync(result, jobs[i].result, IMG_BYTES, cudaMemcpyDeviceToHost, streams[i]));
                return true;
            }
        }
        return false;
    }

    bool dequeue(int *job_id) override
    {
        for (int i = 0; i < STREAM_COUNT; i++) {
            if (jobs[i].job_id == job_context::AVAILABLE) continue;

            cudaError_t status = cudaStreamQuery(streams[i]);
            
            switch (status) {
                case cudaSuccess:
                    *job_id = jobs[i].job_id;
                    jobs[i].job_id = job_context::AVAILABLE;
                    return true;
                case cudaErrorNotReady:
                    break;
                default:
                    CUDA_CHECK(status);
                    return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

/**********************************************************************************************************************/

// TODO implement a SPSC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}

int calc_threadBlock_cnt(int threads) {
    // TODO: BATEL
    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));
    printDevProp(devProp);

    return min(N_IMAGES/QUEUE_SIZE, 10);
}

struct queue {
    struct request {
        uchar *target;
        uchar *reference;

        __device__ __host__ request(): target(new uchar), reference(new uchar) {}
        __device__ __host__ request(uchar *target, uchar *reference): target(target), reference(reference) {}
        __device__ __host__ ~request() { delete target; delete reference; }
    };

    struct response {
        uchar *result;

        __device__ __host__ response(): result(new uchar) {}
        __device__ __host__ response(uchar *result): result(result) {}
        __device__ __host__ ~response() { delete result; }
    };

    job_context jobs[QUEUE_SIZE];
    int index;

    queue(): index(0) {}

    __device__ __host__ bool enqueue_response(response resp) { 
        assert(full() == false);
        jobs[index+1].result = resp.result;
        jobs[index].job_id = 1; // TODO: BATEL
        ++index;
        return true;
    }

    __device__ __host__ request dequeue_request() {
        assert(empty() == false);
        request req(jobs[index].target, jobs[index].reference);
        jobs[index].job_id = job_context::AVAILABLE;
        --index;
        return req;
    }

    bool enqueue_request(request req) {
        assert(full() == false);
        jobs[index+1].target = req.target;
        jobs[index+1].reference = req.reference;
        jobs[index].job_id = 1; // TODO: BATEL
        ++index;
        return true;
    }

    response dequeue_response() {
        assert(empty() == false);
        response resp(jobs[index].result);
        jobs[index].job_id = job_context::AVAILABLE;
        --index;
        return resp;
    }

    __device__ __host__ bool empty() { return index == 0; }

    __device__ __host__ bool full() { return index == QUEUE_SIZE-1; }
};

__device__ void persistent_kernel(queue* cpu_to_gpu_queue, queue* gpu_to_cpu_queue) {
    // TODO: BATEL
    while (1/*running*/) {
        queue::request request;
        queue::response response;

        request = cpu_to_gpu_queue->dequeue_request();
        process_image(request.target, request.reference, response.result);
        gpu_to_cpu_queue->enqueue_response(response);
    }
}

__global__
void process_persistent_kernel(queue* cpu_to_gpu_queues, queue* gpu_to_cpu_queues) {
    persistent_kernel(&cpu_to_gpu_queues[blockIdx.x], &gpu_to_cpu_queues[blockIdx.x]);
}

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
    queue* cpu_to_gpu_queues;
    queue* gpu_to_cpu_queues;
    int queue_cnt;

public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        int queue_cnt = calc_threadBlock_cnt(threads);

        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_queues, queue_cnt * sizeof(queue)));
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_queues, queue_cnt * sizeof(queue)));
        process_persistent_kernel<<<queue_cnt, threads>>>(cpu_to_gpu_queues, gpu_to_cpu_queues);
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_queues));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_queues));
    }

    bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        // TODO push new task into queue if possible
        for (int i = 0; i < queue_cnt; i++) {
            if (cpu_to_gpu_queues[i].full()) continue;
            if (cpu_to_gpu_queues[i].enqueue_request(queue::request(target, reference))) return true;
        }

        return false;
    }

    bool dequeue(int *job_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the job_id of the request that was completed.
        //*job_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
