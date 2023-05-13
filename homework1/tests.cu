#include "ex1.h"
#include <random>
#include "gtest/gtest.h"
#include "randomize_images.h"

#define CUDA_ASSERT(f)             \
    do                             \
    {                              \
        cudaError_t e = f;         \
        ASSERT_EQ(e, cudaSuccess); \
    } while (0)

class Hw1 : public ::testing::Test
{
};

template <typename T>
void randomizeArray(T *arr, int size, int maxVal)
{
    static std::random_device rd;
    static int seed = rd();
    static std::mt19937 gen(seed);
    std::uniform_int_distribution<T> dist(0, maxVal);
    for (int i = 0; i < size; ++i)
    {
        arr[i] = dist(gen);
    }
}

TEST_F(Hw1, Argmin)
{
    // arr represents a row in the "delta" matrix
    int arr[LEVELS];
    int *arrCpy;
    CUDA_ASSERT(cudaHostAlloc(&arrCpy, sizeof(int) * LEVELS, 0));

    for (int i = 0; i < 1000; i++)
    {
        randomizeArray<int>(arr, LEVELS, SIZE * SIZE);
        memcpy(arrCpy, arr, sizeof(int) * LEVELS);
        int cpuArgmin = argmin(arr, LEVELS);
        argminWrapper<<<1, LEVELS / 2>>>(arrCpy, LEVELS);
        cudaDeviceSynchronize();
        int gpuArgmin = arrCpy[1];
        ASSERT_TRUE(cpuArgmin == gpuArgmin);
    }
}

TEST_F(Hw1, ColorHistogram)
{
    uchar *imgBuf;
    CUDA_ASSERT(cudaHostAlloc(&imgBuf, sizeof(uchar) * CHANNELS * SIZE * SIZE, 0));
    int cpuHistograms[CHANNELS][LEVELS];
    int *gpuHistogramsBuf;
    CUDA_ASSERT(cudaHostAlloc(&gpuHistogramsBuf, sizeof(int) * CHANNELS * LEVELS, 0));
    int (*gpuHistograms)[LEVELS]  = (int(*)[LEVELS])gpuHistogramsBuf;

    for (int i = 0; i < 10; i++)
    {
        randomizeImage(imgBuf);
        uchar (*img)[CHANNELS] = (uchar(*)[CHANNELS]) imgBuf;
        colorHist(img, SIZE * SIZE, cpuHistograms);
        colorHistWrapper<<<1, 1024>>>(img, gpuHistograms);
        cudaDeviceSynchronize();

        for (int j = 0; j < CHANNELS; j++)
        {
            for (int k = 0; k < LEVELS; k++){
                ASSERT_EQ(gpuHistograms[j][k], cpuHistograms[j][k])
                    << "i = " << i << ", j = " << j << ", k = " << k << std::endl;
            }
        }
    }
}

TEST_F(Hw1, PrefixSum)
{
    int hist[LEVELS];
    int cpuOut[LEVELS];
    int *gpuOut;
    CUDA_ASSERT(cudaHostAlloc(&gpuOut, sizeof(int) * LEVELS, 0));
    for (int i = 0; i < 100000; i++)
    {
        randomizeArray<int>(hist, LEVELS, SIZE * SIZE);
        memcpy(gpuOut, hist, sizeof(int) * LEVELS);
        prefixSum(hist, LEVELS, cpuOut);
        prefixSumWrapper<<<1, 1024>>>(gpuOut, LEVELS);
        cudaDeviceSynchronize();
        for (int j = 0; j < LEVELS; j++)
        {
            ASSERT_EQ(gpuOut[j], cpuOut[j])
                << "i = " << i << ", j = " << j << std::endl;
        }
    }
}

TEST_F(Hw1, Mapping)
{
    uchar *mapsBuf;
    CUDA_ASSERT(cudaHostAlloc(&mapsBuf, sizeof(int) * LEVELS * CHANNELS, 0));
    uchar *targetImgBuf;
    CUDA_ASSERT(cudaHostAlloc(&targetImgBuf, sizeof(uchar) * SIZE * SIZE * CHANNELS, 0));
    uchar cpuResult[SIZE * SIZE][CHANNELS];
    uchar *gpuResultBuf;
    CUDA_ASSERT(cudaHostAlloc(&gpuResultBuf, sizeof(uchar) * SIZE * SIZE * CHANNELS, 0));
    uchar (*gpuResult)[CHANNELS] = (uchar(*)[CHANNELS])gpuResultBuf;

    for(int i = 0; i < 100; i++){
        randomizeArray<uchar>(mapsBuf, LEVELS * CHANNELS, LEVELS - 1);
        randomizeImage(targetImgBuf);
        uchar (*maps)[LEVELS] = (uchar(*)[LEVELS])mapsBuf;
        uchar (*targetImg)[CHANNELS] = (uchar(*)[CHANNELS])targetImgBuf;
        performMapping(maps, targetImg, cpuResult, SIZE, SIZE);
        performMappingWrapper<<<1, 1024>>>(maps, targetImg, gpuResult);
        cudaDeviceSynchronize();

        for (int j = 0; j < SIZE * SIZE; j++)
        {
            for (int k = 0; k < CHANNELS; k++)
            {
                ASSERT_EQ(gpuResult[j][k], cpuResult[j][k])
                    << "i = " << i << ", j = " << j << ",k = " << k << std::endl;
            }
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();

    cudaDeviceReset();

    return ret;
}
