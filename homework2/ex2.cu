#include "ex2.h"
#include <new>
#include <cuda/atomic>
#include <iostream>

#define QUEUE_SLOTS 16
#define REGISTERS_PER_THREAD 32

#define RUN_IN_QUEUE(code)                  \
    do {                                    \
        for (int i = 0; i < blocks; i++) {  \
            code;                           \
        }                                   \
    } while (0)


//=========================================================================================//
//                             Process Image Kernel Implementation                         //
//=========================================================================================//

__device__ void prefixSum(int arr[], int len, int tid, int threads) {
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

__device__ void argmin(int arr[], int len, int tid, int threads) {
    assert(threads == len / 2);
    int halfLen = len / 2;
    bool firstIteration = true;
    int prevHalfLength = 0;
    while (halfLen > 0) {
        if (tid < halfLen) {
            if (arr[tid] == arr[tid + halfLen]) { // a corner case
                int lhsIdx = tid;
                int rhdIdx = tid + halfLen;
                int lhsOriginalIdx = firstIteration ? lhsIdx : arr[prevHalfLength + lhsIdx];
                int rhsOriginalIdx = firstIteration ? rhdIdx : arr[prevHalfLength + rhdIdx];
                arr[tid + halfLen] = lhsOriginalIdx < rhsOriginalIdx ? lhsOriginalIdx : rhsOriginalIdx;
            }
            else { // the common case
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

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS]) {
    int tid = threadIdx.x;

    if (tid < LEVELS) {
        for (int j = 0; j < CHANNELS; j++) {
            histograms[j][tid] = 0;
        }
    }

    __syncthreads();

    for (int i = tid; i < SIZE * SIZE; i += blockDim.x) {
        for (int j = 0; j < CHANNELS; j++) {
            atomicAdd(histograms[j] + img[i][j], 1);
        }
    }

    __syncthreads();
}

__device__ void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]) {
    int tid = threadIdx.x;

    for (int i = tid; i < SIZE * SIZE; i += blockDim.x) {
        uchar* inRgbPixel = targetImg[i];
        uchar* outRgbPixel = resultImg[i];

        for (int j = 0; j < CHANNELS; j++) {
            uchar* mapChannel = maps[j];
            outRgbPixel[j] = mapChannel[inRgbPixel[j]];
        }
    }
    __syncthreads();
}

__device__ void calculateMap(uchar map[LEVELS], int target[LEVELS], int reference[LEVELS]) {
    __shared__ int diff_row[LEVELS];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for (int i_tar = 0; i_tar < LEVELS; i_tar++) {
        for (int i_ref = tid; i_ref < LEVELS; i_ref += threads) {
            diff_row[tid] = abs(reference[tid] - target[i_tar]);
        }
        __syncthreads();

        argmin(diff_row, LEVELS, tid, LEVELS / 2);
        __syncthreads();

        if (tid == 0) {
            map[i_tar] = diff_row[1];
        }
        __syncthreads();
    }
}

__device__ void process_image(uchar* targets, uchar* refrences, uchar* results) {
    assert(blockDim.x % 2 == 0);

    __shared__ int histograms_tar[CHANNELS][LEVELS];
    __shared__ int histograms_ref[CHANNELS][LEVELS];
    __shared__ uchar map[CHANNELS][LEVELS];

    int tid = threadIdx.x;

    /*Calculate Histograms*/
    colorHist(reinterpret_cast<uchar(*)[CHANNELS]>(targets), histograms_tar);
    colorHist(reinterpret_cast<uchar(*)[CHANNELS]>(refrences), histograms_ref);
    __syncthreads();

    /*Calculate Reference and target Histogram-Prefix sum*/
    for (int k = 0; k < CHANNELS; k++) {
        prefixSum(histograms_tar[k], LEVELS, tid, blockDim.x);
        prefixSum(histograms_ref[k], LEVELS, tid, blockDim.x);
    }
    __syncthreads();

    /*Calculate a map 𝑚 from old to new gray levels*/
    for (int i = 0; i < CHANNELS; i++) {
        calculateMap(map[i], histograms_tar[i], histograms_ref[i]);
    }
    __syncthreads();

    performMapping(map, reinterpret_cast<uchar(*)[CHANNELS]>(targets), reinterpret_cast<uchar(*)[CHANNELS]>(results));
    __syncthreads();
}

__global__ void process_image_kernel(uchar* target, uchar* reference, uchar* result) {
    process_image(target, reference, result);
}


/*Job context struct with necessary CPU / GPU pointers to process a single image*/
struct job_context {
    typedef enum {
        AVAILABLE = -1
    } job_status;

    uchar* target;
    uchar* reference;
    uchar* result;
    int job_id;

    job_context() = default;

    job_context(uchar* target, uchar* reference, uchar* result, int job_id):
        target(target), reference(reference), result(result), job_id(job_id) {}
};

//=========================================================================================//
//                             Stream Server Implementation                                //
//=========================================================================================//

class streams_server : public image_processing_server {
private:
    cudaStream_t streams[STREAM_COUNT];
    job_context jobs[STREAM_COUNT];

public:
    streams_server() {
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&(jobs[i].target), IMG_BYTES * sizeof(uchar)));
            CUDA_CHECK(cudaMalloc(&(jobs[i].reference), IMG_BYTES * sizeof(uchar)));
            CUDA_CHECK(cudaMalloc(&(jobs[i].result), IMG_BYTES * sizeof(uchar)));
            jobs[i].job_id = job_context::AVAILABLE;
        }
    }

    ~streams_server() override {
        for (int i = 0; i < STREAM_COUNT; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(jobs[i].target));
            CUDA_CHECK(cudaFree(jobs[i].reference));
            CUDA_CHECK(cudaFree(jobs[i].result));
        }
    }

    bool enqueue(int job_id, uchar* target, uchar* reference, uchar* result) override {
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

    bool dequeue(int* job_id) override {
        for (int i = 0; i < STREAM_COUNT; i++) {
            if (jobs[i].job_id == job_context::AVAILABLE)
                continue;

            cudaError_t status = cudaStreamQuery(streams[i]);

            switch (status) {
                case cudaSuccess:
                    *job_id = jobs[i].job_id;
                    jobs[i].job_id = job_context::AVAILABLE;
                    return true;
                case cudaErrorNotReady:
                    continue;
                default:
                    CUDA_CHECK(status);
                    return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server() {
    return std::make_unique<streams_server>();
}

//=========================================================================================//
//                           SPSC Queue Server Implementation                              //
//=========================================================================================//

int calculateTBs(int threads) {
    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));
    int registers = threads * 32;
    int shmem_size = ((LEVELS * sizeof(int)) +
        2 * (LEVELS * CHANNELS * sizeof(int)) +
        (LEVELS * CHANNELS * sizeof(uchar)) +
        sizeof(job_context));

    int max_shmem = devProp.sharedMemPerMultiprocessor / shmem_size;
    int max_threads = devProp.maxThreadsPerMultiProcessor / threads;
    int max_registers = devProp.regsPerMultiprocessor / registers;

    int min_threads_shmem = min(max_shmem, max_threads);

    int res = min(max_registers, min_threads_shmem) * devProp.multiProcessorCount;
    return res;
}

template <typename T, uint8_t size>
class ring_buffer {
private:
    static const size_t N = size;
    T _workQueue[N];
    cuda::atomic<size_t> _head, _tail;

public:
    __host__ ring_buffer(): _head(0), _tail(0) {}

    __host__ __device__ bool push(const T& wqe) {
        size_t tail;
        if (full(&tail)) return false;
        _workQueue[tail % N] = wqe;
        _tail.store(tail + 1, cuda::memory_order_release);
        return true;
    }

    __host__ __device__ bool pop(T* wqe) {
        size_t head;
        if (empty(&head)) return false;
        *wqe = _workQueue[head % N];
        _head.store(head + 1, cuda::memory_order_release);
        return true;
    }

    __host__ __device__ bool empty(size_t* head) {
        *head = _head.load(cuda::memory_order_relaxed);
        return (_tail.load(cuda::memory_order_acquire) - *head == 0);
    }

    __host__ __device__ bool full(size_t* tail) {
        *tail = _tail.load(cuda::memory_order_relaxed);
        return (*tail - _head.load(cuda::memory_order_acquire) == N);
    }
};

typedef ring_buffer<job_context, QUEUE_SLOTS> SPSC;

__global__ void process_persistent_kernel(SPSC* cpu_to_gpu_queues, SPSC* gpu_to_cpu_queues, bool* running) {
    __shared__ job_context request;

    while (*running) {
        if (threadIdx.x == 0) {
            if (!(cpu_to_gpu_queues[blockIdx.x].pop(&request))) continue;
        }
        __syncthreads();

        process_image(request.target, request.reference, request.result);
        __syncthreads();

        if (threadIdx.x == 0) {
            while(!(gpu_to_cpu_queues[blockIdx.x].push(request)));
        }
        __syncthreads();
    }
}

class queue_server : public image_processing_server {
private:
    int blocks;
    bool* running;

    SPSC* cpu_to_gpu_queues;
    SPSC* gpu_to_cpu_queues;

public:
    queue_server(int threads) {
        blocks = calculateTBs(threads);

        CUDA_CHECK(cudaMallocHost(&running, sizeof(bool)));
        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_queues, blocks * sizeof(SPSC)));
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_queues, blocks * sizeof(SPSC)));

        ::new (running) bool(true);

        RUN_IN_QUEUE(
            ::new (&cpu_to_gpu_queues[i]) SPSC();
            ::new (&gpu_to_cpu_queues[i]) SPSC();
        );

        process_persistent_kernel<<<blocks, threads>>>(cpu_to_gpu_queues, gpu_to_cpu_queues, running);
    }

    ~queue_server() override {
        // Kills the server
        *running = false;

        // Free resources allocated in constructor
        RUN_IN_QUEUE(
            gpu_to_cpu_queues[i].~SPSC();
            cpu_to_gpu_queues[i].~SPSC();
        );

        CUDA_CHECK(cudaFreeHost(running));
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_queues));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_queues));
    }

    bool dequeue(int* job_id) override {
        job_context job;

        // Return the job_id of the request that was completed
        RUN_IN_QUEUE(
            if (!(gpu_to_cpu_queues[i].pop(&job))) continue;
            *job_id = job.job_id;
            return true;
        );

        return false;
    }

    bool enqueue(int job_id, uchar* target, uchar* reference, uchar* result) override {
        job_context new_job(target, reference, result, job_id);
        return cpu_to_gpu_queues[job_id % blocks].push(new_job);
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads) {
    return std::make_unique<queue_server>(threads);
}
