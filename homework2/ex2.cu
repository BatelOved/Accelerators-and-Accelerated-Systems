#include "ex2.h"
#include <cuda/atomic>

#define KILL_JOB_ID (-2)
#define QUEUE_SLOTS 4
#define REGISTERS_PER_THREAD 32

#define RUN_IN_QUEUE(code)      \
    do                          \
    {                           \
        int i = 0;              \
        while (i < queue_count) \
        {                       \
            code                \
                i++;            \
        }                       \
    } while (0)

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
        uchar* rgbPixel = img[i];
        for (int j = 0; j < CHANNELS; j++) {
            int* channelHist = histograms[j];
            atomicAdd(&channelHist[rgbPixel[j]], 1);
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

/**********************************************************************************************************************/

/*Job context struct with necessary CPU / GPU pointers to process a single image*/
typedef struct {
    typedef enum {
        AVAILABLE = -1
    } job_status;

    uchar* target;
    uchar* reference;
    uchar* result;
    int job_id;
} job_context;

__global__ void process_image_kernel(uchar* target, uchar* reference, uchar* result) {
    process_image(target, reference, result);
}

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

/*********************** Stream server implementation - end ***********************/

/**********************************************************************************/
/*************************** SPSC Queue implementation ****************************/
/**********************************************************************************/

int calc_threadBlock_cnt(int threads) {
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


template <typename T, uint8_t size> class ring_buffer {
private:
    static const size_t N = 1 << size;
    T _jobs[N];
    cuda::atomic<size_t> _head = 0, _tail = 0;

public:
    void push(const T& job) {
        int tail = _tail.load(cuda::memory_order_relaxed);
        while (tail - _head.load(cuda::memory_order_acquire) == N);
        _jobs[_tail % N] = job;
        _tail.store(tail + 1, cuda::memory_order_release);
    }

    T pop() {
        int head = _head.load(cuda::memory_order_relaxed);
        while (_tail.load(cuda::memory_order_acquire) == _head);
        T item = _jobs[_head % N];
        _head.store(head + 1, cuda::memory_order_release);
        return item;
    }
};

class queue {
private:
    size_t N;
    job_context* jobs;
    cuda::atomic<size_t> queue_head, queue_tail;

public:
    __host__ queue() : queue_head(0), queue_tail(0), N(QUEUE_SLOTS) {
        CUDA_CHECK(cudaMallocHost(&jobs, N * sizeof(job_context)));
    }

    __host__ ~queue() {
        CUDA_CHECK(cudaFreeHost(jobs));
    }

    __device__ __host__ job_context dequeue_request() {
        job_context item;
        item.job_id = -1;
        int head = queue_head.load(cuda::memory_order_relaxed);
        if (queue_tail.load(cuda::memory_order_acquire) != queue_head) {
            item = jobs[queue_head % N];
            jobs[queue_head % N].job_id = -1;
            queue_head.store(head + 1, cuda::memory_order_release);
        }
        return item;
    }

    __device__ __host__ bool enqueue_response(const job_context& new_job) {
        int tail = queue_tail.load(cuda::memory_order_relaxed);
        if (N == (tail - queue_head.load(cuda::memory_order_acquire))) {
            return false;
        }
        jobs[queue_tail % N] = new_job;
        queue_tail.store(tail + 1, cuda::memory_order_release);
        return true;
    }
};

__global__ void process_persistent_kernel(queue* cpu_to_gpu_queues, queue* gpu_to_cpu_queues) {
    __shared__ job_context request;

    request.job_id = -1;
    int thread = threadIdx.x;

    while (true) {
        if (thread == 0)
            request = cpu_to_gpu_queues[blockIdx.x].dequeue_request();
        __syncthreads();

        if (request.job_id == KILL_JOB_ID)
            break;

        if (request.job_id != -1) {
            process_image(request.target, request.reference, request.result);
            __syncthreads();

            if (thread == 0)
                while (!gpu_to_cpu_queues[blockIdx.x].enqueue_response(request))
                    ;
            __syncthreads();
        }
    }
}

class queue_server : public image_processing_server {
private:
    int queue_count;
    int curr_block;

    queue* cpu_to_gpu_queues;
    char* buffer_CPU_GPU;

    queue* gpu_to_cpu_queues;
    char* buffer_GPU_CPU;

public:
    queue_server(int threads) {
        queue_count = calc_threadBlock_cnt(threads);
        curr_block = 0;

        CUDA_CHECK(cudaMallocHost(&buffer_GPU_CPU, queue_count * sizeof(queue)));
        gpu_to_cpu_queues = reinterpret_cast<queue*>(buffer_GPU_CPU);

        CUDA_CHECK(cudaMallocHost(&buffer_CPU_GPU, queue_count * sizeof(queue)));
        cpu_to_gpu_queues = reinterpret_cast<queue*>(buffer_CPU_GPU);

        RUN_IN_QUEUE(new (&gpu_to_cpu_queues[i]) queue();
        new (&cpu_to_gpu_queues[i]) queue(););

        process_persistent_kernel<<<queue_count, threads>>>(cpu_to_gpu_queues, gpu_to_cpu_queues);
    }

    ~queue_server() override {
        // Sends kill signal to the threads
        RUN_IN_QUEUE(this->enqueue(KILL_JOB_ID, nullptr, nullptr, nullptr););

        // Free resources allocated in constructor
        RUN_IN_QUEUE(gpu_to_cpu_queues[i].~queue();
        cpu_to_gpu_queues[i].~queue(););
        CUDA_CHECK(cudaFreeHost(buffer_GPU_CPU));
        CUDA_CHECK(cudaFreeHost(buffer_CPU_GPU));
    }

    bool dequeue(int* job_id) override {
        job_context request;
        request.job_id = -1;

        RUN_IN_QUEUE(request = gpu_to_cpu_queues[i].dequeue_request();
        if (request.job_id != -1) {
            *job_id = request.job_id;
            return true;
        });

        return false;
    }

    bool enqueue(int job_id, uchar* target, uchar* reference, uchar* result) override {
        job_context new_job;
        new_job.reference = reference;
        new_job.target = target;
        new_job.job_id = job_id;
        new_job.result = result;

        bool res = cpu_to_gpu_queues[curr_block].enqueue_response(new_job);

        if (res) {
            curr_block = (curr_block + 1);
            curr_block = curr_block % queue_count;
        }
        return res;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads) {
    return std::make_unique<queue_server>(threads);
}
