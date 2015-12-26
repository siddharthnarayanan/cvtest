#include <string>
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;


//----------------------------------------------------------
// Histogram Comparison
//----------------------------------------------------------
void histogramcompare(Mat src_base,Mat src_test1)
{	
	Mat hsv_base, hsv_test1;
	cvtColor( src_base, hsv_base, COLOR_GRAY2BGR );
	cvtColor( hsv_base, hsv_base, COLOR_BGR2HSV );
	cvtColor( src_test1, hsv_test1, COLOR_GRAY2BGR );
	cvtColor( hsv_test1, hsv_test1, COLOR_BGR2HSV );

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };

	/// Histograms
	MatND hist_base;
	MatND hist_test1;

	/// Calculate the histograms for the HSV images
	calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
	normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

	calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
	normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

	/// Apply the histogram comparison methods
	for( int i = 0; i < 1; i++ )
	{
		int compare_method = i;
		double base_base = compareHist( hist_base, hist_base, compare_method );
		double base_test1 = compareHist( hist_base, hist_test1, compare_method );

		printf( " Method [%d] Perfect, Base-Test(1) : %f \n\n", i, base_test1);
	}
}
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
	float intensityhigh=0,intensitylow=1.0,tmag=0;
	// src image
	Mat img1,img2,MagCopy1,MagCopy2;
	// Magnitude
	Mat Mag1,Mag2;
	// Phase
	Mat Phase1,Phase2;
	// Image loading
	img1=imread("f:\\ImagesForTest\\pic1.jpg",0);
	resize(img1,img1,Size(512,512));
	img2=imread("f:\\ImagesForTest\\pic2.jpg",0);
	resize(img2,img2,Size(512,512));

	// Image size
	cout<<"Image-1 : Width :"<<img1.size().width<<endl;
	cout<<"Image-1 : Height :"<<img1.size().height<<endl;
	cout<<"Image-2 : Width :"<<img2.size().width<<endl;
	cout<<"Image-2 : Height :"<<img2.size().height<<endl;

	//----------------------------------------------------------
	// Fourier transform
	//----------------------------------------------------------
	ForwardFFT_Mag_Phase(img1,Mag1,Phase1);   
	ForwardFFT_Mag_Phase(img2,Mag2,Phase2);   
	//----------------------------------------------------------
	// Filter
	//----------------------------------------------------------
	// draw ring    
	int R=20;  // External radius
	int r=0;   // internal radius
	float value; //intensity value
	float intensityavgimg1, intensityavgimg2; //average intensity value array for images
	float intensityimg1[512]; //intensity value array for image 1
	float intensityimg2[512]; //intensity value array for image 2
	/*********************float intensityimg[512][512]; //intensity value array for image 1******************/	
	
	Mat mask1,mask2;
	mask1.create(Mag1.cols,Mag1.rows,CV_32F);
	mask2.create(Mag2.cols,Mag2.rows,CV_32F);
	int cx = Mag1.cols>>1;
	int cy = Mag1.rows>>1;       
	mask1=1,mask2=1;
	cv::circle(mask1,cv::Point(cx,cy),R,CV_RGB(0,0,0),-1);   
	cv::circle(mask1,cv::Point(cx,cy),r,CV_RGB(1,1,1),-1);
	cv::circle(mask2,cv::Point(cx,cy),R,CV_RGB(0,0,0),-1);   
	cv::circle(mask2,cv::Point(cx,cy),r,CV_RGB(1,1,1),-1);
	//mask=1-mask; // comment for low pass filter

	cv::multiply(Mag1,mask1,Mag1); // comment to turn filter off for magnitude
	cv::multiply(Mag2,mask2,Mag2); // comment to turn filter off for magnitude
	//cv::multiply(Phase,mask,Phase); // comment to turn filter off for phase
	//----------------------------------------------------------
	// Inverse transform
	//----------------------------------------------------------
	InverseFFT_Mag_Phase(Mag1,Phase1,img1);    
	InverseFFT_Mag_Phase(Mag2,Phase2,img2);
	//----------------------------------------------------------
	// Results output
	//----------------------------------------------------------
	// 
	Mat LogMag1;
	LogMag1.zeros(Mag1.rows,Mag1.cols,CV_32F);
	LogMag1=(Mag1+1);
	cv::log(LogMag1,LogMag1);
	Mat LogMag2;
	LogMag2.zeros(Mag2.rows,Mag2.cols,CV_32F);
	LogMag2=(Mag2+1);
	cv::log(LogMag2,LogMag2);
	//---------------------------------------------------

	imshow("Image-1: Magnitude Log", LogMag1);
	imshow("Image-2: Magnitude Log", LogMag2);
	LogMag1.copyTo(MagCopy1);
	LogMag2.copyTo(MagCopy2);
	//MagCopy.convertTo(MagCopy,CV_8UC1);
	imshow("Image-1: Fourier transfom Filtered result", MagCopy1);
	imshow("Image-1: Phase", Phase1);
	imshow("Image-2: Fourier transfom Filtered result", MagCopy2);
	imshow("Image-2: Phase", Phase2);
	// img1 - now in CV_32FC1 format,we need CV_8UC1 or scale it by factor 1.0/255.0  
	img1.convertTo(img1,CV_8UC1);
	imshow("Image-1: Filtered result", img1);    
	img2.convertTo(img2,CV_8UC1);
	imshow("Image-2: Filtered result", img2);  

	intensityhigh=0,intensitylow=1.0,tmag=0;
	for (int y=0;y<LogMag1.rows;y++)
	{for (int x=0;x<LogMag1.cols;x++)
	{
		value = LogMag1.at<float>(y,x);
		intensityimg1[y] = value;
		tmag+=value;
		if( intensityhigh < (float)value)
			intensityhigh=(float)value;
		if( intensitylow > (float)value)
			intensitylow=(float)value;

	}
	} 
	intensityavgimg1=tmag/(512*512);
	cout<<"________________________________________________"<<"\n\n";
	cout << "Image-1: Intensity value of last pixel of FFT-Magnitude is :"<<value<<"\n";
	cout << "Image-1: Intensity average value of FFT-Magnitude is :"<<intensityavgimg1<<"\n";
	cout << "Image-1: Intensity high value of FFT-Magnitude is :"<<intensityhigh<<"\n";
	cout << "Image-1: Intensity low value of FFT-Magnitude is :"<<intensitylow<<"\n";
	cout << "Image-1: Intensity total Magnitude of FFT-Magnitude is :"<<tmag<<"\n";

	intensityhigh=0,intensitylow=1.0,tmag=0;
	for (int y=0;y<LogMag2.rows;y++)
	{for (int x=0;x<LogMag2.cols;x++)
	{
		value = LogMag2.at<float>(y,x);
		intensityimg2[y] = value;
		tmag+=value;
		if( intensityhigh < (float)value)
			intensityhigh=(float)value;
		if( intensitylow > (float)value)
			intensitylow=(float)value;

	}
	}  
	intensityavgimg2=tmag/(512*512);
	cout<<"________________________________________________"<<"\n\n";
	cout << "Image-2: Intensity value of last pixel of FFT-Magnitude is :"<<value<<"\n";
	cout << "Image-2: Intensity average value of FFT-Magnitude is :"<<intensityavgimg2<<"\n";
	cout << "Image-2: Intensity high value of FFT-Magnitude is :"<<intensityhigh<<"\n";
	cout << "Image-2: Intensity low value of FFT-Magnitude is :"<<intensitylow<<"\n";
	cout << "Image-2: Intensity total Magnitude of FFT-Magnitude is :"<<tmag<<"\n";

	//----------------------------------------------------------
	// Similarity Formula
	//----------------------------------------------------------

	float intensitysum=0;
	for (int y=0;y<LogMag2.rows;y++)
	{for (int x=0;x<LogMag2.cols;x++)
	{
		intensitysum += intensityimg1[y] * intensityimg2[y];

	}
	}

	cout<<"________________________________________________"<<"\n\n";
	cout << "Cumulative: Sum of Intensity values of FFT-Magnitude of both images is :"<<intensitysum<<"\n";


	//----------------------------------------------------------
	// Wait key press
	//----------------------------------------------------------
	waitKey(0);
	return 0;
}
