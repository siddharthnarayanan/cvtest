#include "opencv2/highgui/highgui.hpp"
#include <iostream>
 
using namespace std;
using namespace cv;
 
int main(int argc, char ** argv)
{
    
    const char* filename = argc >=2 ? argv[1] : "pic.jpg";

    Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if( img.empty())
	{
        cout << "no image found";
		return -1;
	}

    //Mat img = imread("pic.JPG",0);
     
    Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
    Mat magI,imI,complexI;    //Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
    merge(planes, 2, complexI);
    dft(complexI, complexI);  // Applying DFT
	split( complexI,planes);
	magnitude(planes[0],planes[1],magI);
	phase(planes[0],planes[1],imI);

	magI += Scalar::all(1);
	log(magI,magI);

    // Reconstructing original imae from the DFT coefficients
    Mat invDFT, invDFTcvt;
    idft(complexI, invDFT, DFT_SCALE | DFT_REAL_OUTPUT ); // Applying IDFT
    invDFT.convertTo(invDFTcvt, CV_8U); 
    
     //show the image
    imshow("Output", invDFTcvt);
	imshow("Original Image", img);
	imshow("Mag Output",magI);
	imshow("Imaginary",imI);
     
    // Wait until user press some key
    waitKey(0);
    return 0;
}
