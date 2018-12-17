//C++ Includes
#include <stdexcept>
#include <iostream>

//OpenCV Includes
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//SPV Includes
//#include "OPENCV_GPUBilateralFilter.h"
#include "../include/Bilateral_kernel.h"

using std::cout;
using std::endl;
using std::string;
using std::logic_error;
using cv::Mat;
using cv::imread;
using cv::imwrite;
using namespace cv;


int main(int argc, char **argv){

 	string source="test-crop-512.jpg";
	Mat img = imread(source);

	if(!img.data)
		return -1;

	// Remove Green and Blue channels
//    Mat channel[3];
//    split(img, channel);
//	channel[2]=Mat::zeros(img.rows, img.cols, CV_8UC1);
//	merge(channel,3,img);
//	channel[1]=Mat::zeros(img.rows, img.cols, CV_8UC1);
//	merge(channel,3,img);


	int t = 0;
	int r = 7 ;
	double gs = 12.0;
	double gr = 16.0;

	string::size_type pAt=source.find_last_of('.');
	int raio=2*r+1;
	string NAME, mask=std::to_string(raio)+"x"+std::to_string(raio);

	CUDABilateralFilter *p;
	p=new CUDABilateralFilter(r, gs, gr);

	Mat dst(img.size(), img.type());
	p->apply(img, dst);

//    namedWindow("window", WINDOW_AUTOSIZE );// Create a window for display.
//	imshow("window", dst);
//	waitKey(0);

	cv::imwrite("image-output.jpg", dst);

	return 0;
}
