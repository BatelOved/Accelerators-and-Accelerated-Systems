///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <memory>

#include "randomize_images.h"

#define N_IMAGES 10000
#define SIZE 32
#define CHANNELS 3
#define LEVELS 256
#define IMG_BYTES (CHANNELS * SIZE * SIZE)

#define STREAM_COUNT 64

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#ifndef DEBUG
#define dbg_printf(...)
#else
#define dbg_printf(...) do { printf(__VA_ARGS__); } while (0)
#endif

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}


/* Abstract base class for both parts of the exercise */
class image_processing_server
{
public:
    virtual ~image_processing_server() {}

    /* Enqueue a pair of images (target and reference) for processing. Receives pointers to pinned host
     * memory. Return false if there is no room for image (caller will try again).
     */
    virtual bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) = 0;

    /* Checks whether any pair of images has completed processing. If so, set job_id
     * accordingly, and return true. */
    virtual bool dequeue(int *job_id) = 0;
};

std::unique_ptr<image_processing_server> create_streams_server();
std::unique_ptr<image_processing_server> create_queues_server(int threads);

///////////////////////////////////////////////////////////////////////////////////////////////////////////

