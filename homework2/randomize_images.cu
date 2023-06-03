#include "randomize_images.h"

void smapleFromPiecewiseLinearDistribution(uchar *outputArr, int len, float width, float shift) {

    //Initialize a random number generator
    static thread_local std::random_device rd;
    static thread_local int seed = rd();
    static thread_local std::minstd_rand gen(seed);

    // Define a piecewise_linear distribution
    float widthLevels = width * UCHAR_MAX;
    float shiftLevels = (UCHAR_MAX - widthLevels) * shift;
    float x[4] = {0, shiftLevels, shiftLevels + widthLevels, UCHAR_MAX};
    float w[4] = {0, 1,  1,  0};
    std::piecewise_linear_distribution<float> dist(x, x+4, w);

    // Generate len random numbers within the desired range
    for(int i = 0; i < len; i++){
        // Round the generated value to the nearest integer and add it to the array
        outputArr[i] = (uchar)std::lround(dist(gen));
    }
}

void randomizeImage(uchar *img){
    static std::random_device rd;
    static int seed = rd();
    static std::minstd_rand gen(seed);
    static std::uniform_real_distribution<float> distribution(0, 1);
    for(int i = 0; i <  CHANNELS; i++){
        float width = distribution(gen);        
        float shift = distribution(gen);
        uchar* channel = img + i * SIZE * SIZE;
        smapleFromPiecewiseLinearDistribution(channel, SIZE * SIZE, width, shift);
    }
}


void randomizeImages(uchar *images)
{
    for(int i = 0; i < N_IMAGES; i++){
        uchar* img = images + i * IMG_BYTES;
        randomizeImage(img);
    }
} 