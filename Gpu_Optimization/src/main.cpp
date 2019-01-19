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

 	string source="crop-128.jpg";
	Mat img = imread(source);

	if(!img.data)
		return -1;

	int t = 0;
	int r = 7;
	double gs = 12.0;
	double gr = 16.0;

	string::size_type pAt=source.find_last_of('.');
	int raio=2*r+1;
	string NAME, mask=std::to_string(raio)+"x"+std::to_string(raio);

	CUDABilateralFilter *p;
	p=new CUDABilateralFilter(r, gs, gr);

	Mat dst(img.size(), img.type());
	p->apply(img, dst);

	cv::imwrite("output-128.jpg", dst);

	return 0;
}
