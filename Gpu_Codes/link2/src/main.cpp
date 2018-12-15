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

//void Process(Mat src, ProcessingInterface *p, string name){
//	Mat dst(src.size(), src.type());
//	p->apply(src, dst);
////    printf ("%s \n", name);
//    namedWindow("window", WINDOW_AUTOSIZE );// Create a window for display.
//	imshow("window", dst);
//	waitKey( 0);
//	cv::imwrite("/home/halil/Hcttp_Msc/CMP_674_GPU_Programming/github/Bilateral_Filtering_Cuda_CMP674/Gpu_Codes/link2/image-output.jpg", dst);
//
//}

int main(int argc, char **argv){

 	string source="test-crop-512.jpg";
	Mat img = imread(source);
//	Mat img(img_c.size(), CV_8UC1);
//	cv::cvtColor(img_c, img, COLOR_RGB2GRAY);
//	cv::imwrite("/home/halil/Hcttp_Msc/CMP_674_GPU_Programming/github/Bilateral_Filtering_Cuda_CMP674/Gpu_Codes/link2/image.png", img);

	if(!img.data)
		return -1;

	int t = 0;
	int r = 5 ;
	double gs = 12.0;
	double gr = 16.0;

	string::size_type pAt=source.find_last_of('.');
	int raio=2*r+1;
	string NAME, mask=std::to_string(raio)+"x"+std::to_string(raio);

	CUDABilateralFilter *p;
	
	p=new CUDABilateralFilter(r, gs, gr);

	Mat dst(img.size(), img.type());
	p->apply(img, dst);

    namedWindow("window", WINDOW_AUTOSIZE );// Create a window for display.
	imshow("window", dst);
	waitKey( 0);
	cv::imwrite("image-output.jpg", dst);


//	Process(img, p, NAME);
	return 0;
}
