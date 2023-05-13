///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex1.h"
#include "randomize_images.h"

#include <random>

#define SQR(a) ((a) * (a))

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    bool foundMistake = false;
    for (int i = 0; i < N_IMAGES * CHANNELS * SIZE * SIZE; i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
        if(!foundMistake && (img_arr1[i] - img_arr2[i] != 0)){
            //printf("%d: %d - %d = %d\n",i, img_arr1[i], img_arr2[i],  img_arr1[i] - img_arr2[i]);
            foundMistake = true;
        }
    }
    return distance_sqr;
}



int main() {
    uchar *images_target;
    uchar *images_refrence;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    int devices;
    CUDA_CHECK( cudaGetDeviceCount(&devices) );
    printf("Number of devices: %d\n", devices);

    CUDA_CHECK( cudaHostAlloc(&images_target, N_IMAGES * SIZE * SIZE * CHANNELS, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_refrence, N_IMAGES * SIZE * SIZE * CHANNELS, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * SIZE * SIZE * CHANNELS, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * SIZE * SIZE * CHANNELS, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * SIZE * SIZE * CHANNELS, 0) );

    double t_start, t_finish;

    /* instead of loading real images, we'll load the arrays with random data */
    printf("\n=== Randomizing images ===\n");
    t_start = get_time_msec();
    randomizeImages(images_target);
    randomizeImages(images_refrence);
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_target = &images_target[i * SIZE * SIZE * CHANNELS];
        uchar *img_refrence = &images_refrence[i * SIZE * SIZE * CHANNELS];
        uchar *img_out = &images_out_cpu[i * SIZE * SIZE * CHANNELS];
        cpu_process((uchar(*)[CHANNELS]) img_target, (uchar(*)[CHANNELS]) img_refrence, (uchar(*)[CHANNELS]) img_out, SIZE, SIZE);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n");

    struct task_serial_context *ts_context = task_serial_init();
    
    CUDA_CHECK(cudaDeviceSynchronize());
    t_start = get_time_msec();
    task_serial_process(ts_context, images_target, images_refrence, images_out_gpu_serial);
    t_finish = get_time_msec();
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial);
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr);

    task_serial_free(ts_context);

    // GPU bulk
    printf("\n=== GPU Bulk ===\n");
    struct gpu_bulk_context *gb_context = gpu_bulk_init();
    CUDA_CHECK(cudaDeviceSynchronize());
    t_start = get_time_msec();
    gpu_bulk_process(gb_context, images_target, images_refrence, images_out_gpu_bulk);
    t_finish = get_time_msec();
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk);
    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr);

    gpu_bulk_free(gb_context);

    return 0;
}
