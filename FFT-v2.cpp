#include <string>
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
//----------------------------------------------------------
// Recombinate quadrants
//----------------------------------------------------------
void Recomb(Mat &src,Mat &dst)
{
    int cx = src.cols>>1;
    int cy = src.rows>>1;
    Mat tmp;
    tmp.create(src.size(),src.type());
    src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
    src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));  
    src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
    src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
    dst=tmp;
}
//----------------------------------------------------------
// Forward fft
//----------------------------------------------------------
void ForwardFFT(Mat &Src, Mat *FImg)
{
    int M = getOptimalDFTSize( Src.rows );
    int N = getOptimalDFTSize( Src.cols );
    Mat padded;    
    copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
    // Create complex image
    // planes[0] image , planes[1] filled by zeroes
    Mat planes[2] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg); 
    dft(complexImg, complexImg,DFT_SCALE);    
    // After tranform we also get complex image 
    split(complexImg, planes);

    // 
    planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
    planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));

    Recomb(planes[0],planes[0]);
    Recomb(planes[1],planes[1]);

    FImg[0]=planes[0].clone();
    FImg[1]=planes[1].clone();

	imshow("Original Image Mag", FImg[0]); 
	imshow("Original Image Phase", FImg[1]); 
}
//----------------------------------------------------------
// Inverse FFT
//----------------------------------------------------------
void InverseFFT(Mat *FImg,Mat &Dst)
{
    Recomb(FImg[0],FImg[0]);
    Recomb(FImg[1],FImg[1]);
    Mat complexImg;
    merge(FImg, 2, complexImg);
    // Inverse transform
    dft(complexImg, complexImg,  DFT_INVERSE);
    split(complexImg, FImg);        
    FImg[0].copyTo(Dst);
}
//----------------------------------------------------------
// Forward FFT using Magnitude and phase
//----------------------------------------------------------
void ForwardFFT_Mag_Phase(Mat &src, Mat &Mag,Mat &Phase)
{
    Mat planes[2];
    ForwardFFT(src,planes);
    Mag.zeros(planes[0].rows,planes[0].cols,CV_32F);
    Phase.zeros(planes[0].rows,planes[0].cols,CV_32F);
    cv::cartToPolar(planes[0],planes[1],Mag,Phase);
}
//----------------------------------------------------------
// Inverse FFT using Magnitude and phase
//----------------------------------------------------------
void InverseFFT_Mag_Phase(Mat &Mag,Mat &Phase, Mat &dst)
{
    Mat planes[2];
    planes[0].create(Mag.rows,Mag.cols,CV_32F);
    planes[1].create(Mag.rows,Mag.cols,CV_32F);
    cv::polarToCart(Mag,Phase,planes[0],planes[1]);
    InverseFFT(planes,dst);
}
//----------------------------------------------------------
// MAIN
//----------------------------------------------------------
int main(int argc, char* argv[])
{
    // src image
    Mat img;
    // Magnitude
    Mat Mag;
    // Phase
    Mat Phase;
    // Image loading (grayscale)
    img=imread("d:\\ImagesForTest\\lena.jpg",0);
    ForwardFFT_Mag_Phase(img,Mag,Phase); 
	imshow("Original Image ", img); 
    //----------------------------------------------------------
    // Inverse transform
    //----------------------------------------------------------
    InverseFFT_Mag_Phase(Mag,Phase,img);    
    img.convertTo(img,CV_8UC1);
	imshow("Filtering result", img);    
    //----------------------------------------------------------
    // Wait key press
    //----------------------------------------------------------
    waitKey(0);
    return 0;
}
