///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <tuple>
#include <assert.h>

#define SIZE 128
#define N_IMAGES 10000

#define LEVELS 256
#define CHANNELS 3


typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

void cpu_process(uchar targetImg[][CHANNELS], uchar refrenceImg[][CHANNELS],  uchar outputImg[][CHANNELS], int width, int height);

struct task_serial_context;

struct task_serial_context* task_serial_init();
void task_serial_process(struct task_serial_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result);
void task_serial_free(struct task_serial_context *context);

struct gpu_bulk_context;

struct gpu_bulk_context *gpu_bulk_init();
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result);
void gpu_bulk_free(struct gpu_bulk_context *context);

#ifdef UNIT_TEST
//whitebox testing requires defenitions of internal functions:

    int argmin(int arr[], int size);
    __global__ void argminWrapper(int *arr, int size);

    void colorHist(uchar img[][CHANNELS], int pixelCount, int histograms[][LEVELS]);
    __global__ void colorHistWrapper(uchar img[][CHANNELS], int histograms[][LEVELS]);

    void prefixSum(int arr[], int size, int res[]);
    __global__ void prefixSumWrapper(int arr[], int size);

    void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS], int width, int height);
    __global__ void performMappingWrapper(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]);

#endif


///////////////////////////////////////////////////////////////////////////////////////////////////////////


