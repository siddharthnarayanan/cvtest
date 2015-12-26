#include "opencv2/highgui/highgui.hpp"
#include <iostream>
 
using namespace std;
using namespace cv;
 
int main()
{
    // Read image from file
    // Make sure that the image is in grayscale
    Mat img = imread("pic.JPG",0);
 
    Mat dftInput1, dftImage1, inverseDFT, inverseDFTconverted;
    img.convertTo(dftInput1, CV_32F);
    dft(dftInput1, dftImage1, DFT_COMPLEX_OUTPUT);    // Applying DFT
 
    // Reconstructing original imae from the DFT coefficients
    idft(dftImage1, inverseDFT, DFT_SCALE | DFT_REAL_OUTPUT ); // Applying IDFT
    inverseDFT.convertTo(inverseDFTconverted, CV_8U);
    imshow("Output", inverseDFTconverted);
 
    //show the image
    imshow("Original Image", img);
     
    // Wait until user press some key
    waitKey(0);
    return 0;
}
