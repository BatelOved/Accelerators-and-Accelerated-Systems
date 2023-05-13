///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex1.h"
// #define RUN_ON_GPU

//#include <opencv2/opencv.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int argc,char **argv) 
{ 
    char *targetImageName = argv[1];
    Mat targetImage = imread(targetImageName, IMREAD_COLOR);
    //cvtColor(targetImage, targetImage, CV_BGR2RGB);

    char *refrenceImageName = argv[2];
    Mat refrenceImage = imread(refrenceImageName, IMREAD_COLOR);
    //cvtColor(refrenceImage, refrenceImage, CV_BGR2RGB);

    if( argc != 3 || !targetImage.data || !refrenceImage.data)
    {
        printf( " No images data \n " );
        return 1;
    }

    // Mat grayTarget;
    // cvtColor(targetImage, grayTarget, CV_BGR2GRAY);
    // Mat grayRefrence;
    // cvtColor(refrenceImage, grayRefrence, CV_BGR2GRAY);
    // imwrite("grayscale.png", grayTarget);

    Size sz = targetImage.size();
    assert(refrenceImage.size() == sz);
    assert(targetImage.isContinuous());
    assert(refrenceImage.isContinuous());

#ifndef RUN_ON_GPU
    uchar *outputImg = new uchar[CHANNELS * sz.width * sz.height];
    cpu_process((uchar(*)[CHANNELS])targetImage.data, (uchar(*)[CHANNELS])refrenceImage.data, (uchar(*)[CHANNELS])outputImg, sz.width, sz.height);
    Mat result(sz, CV_8UC3, outputImg);
#else
    assert((SIZE <= sz.height) && (SIZE <= sz.width));
    Mat croppedTarget = targetImage(Rect(0,0,SIZE,SIZE)).clone();
    imwrite("images/croppedTarget.bmp", croppedTarget);
    Mat croppedRefrence = refrenceImage(Rect(0,0,SIZE,SIZE)).clone();
    imwrite("images/croppedRefrence.bmp", croppedRefrence);
    uchar *outputImg = new uchar[CHANNELS * SIZE * SIZE];
    assert(N_IMAGES == 1);
    task_serial_context *context = task_serial_init();
    task_serial_process(context , croppedTarget.data, croppedRefrence.data, outputImg);
    task_serial_free(context);
    Mat result(croppedTarget.size(), CV_8UC3, outputImg);
#endif
    
    imwrite("images/result.bmp", result);
    printf("result iamge was saved in images/result.bmp\n");

    delete outputImg;

    return 0;
}
