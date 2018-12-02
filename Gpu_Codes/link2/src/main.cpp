//C++ Includes
#include <stdexcept>
#include <iostream>

//OpenCV Includes
#include <opencv2/highgui/highgui.hpp>

//SPV Includes
//#include "OPENCV_GPUBilateralFilter.h"
#include "Bilateral_kernel.h"

using std::cout;
using std::endl;
using std::string;
using std::logic_error;
using cv::Mat;
using cv::imread;
using cv::imwrite;

static void help(){
	cout << "Usage:\n"
	"./CUDA-Bilateral-Filter [image_name] " <<
	"[Implementation (0 - OpenCV GPU, others - CUDA] " <<
	"[radius (1-50)] [Spatial Gaussian (>1)] [Color Gaussian (>1)]\n" << endl;
}

void Process(Mat src, ProcessingInterface *p, string name){
	Mat dst(src.size(), src.type());
	p->apply(src, dst);

	imwrite(name, dst);
}

int main(int argc, char **argv){
	//help();
	//if(argc!=6){
	//	help();
	//	return -1;
	//}
	string source="image-13.jpg";
	Mat img=imread(source);
	if(!img.data)
		return -1;

	int t = 0;//atoi(argv[2]), r=atoi(argv[3]);
	//if(r<1 && r>50)
	//	return -1;
	int r = 5;
	double gs = 12.0;
	double gr = 16.0;

	//double gs=atoi(argv[4]), gr=atoi(argv[5]);
	//if(gs<1 || gr<1)
	//	return -1;

	string::size_type pAt=source.find_last_of('.');
	int raio=2*r+1;
	string NAME, mask=std::to_string(raio)+"x"+std::to_string(raio);

	ProcessingInterface *p;
	NAME=source.substr(0, pAt)+"Bilateral"+mask+".jpg";
	
		//p=new OPENCV_GPUBilateralFilter(r, gs, gr);
	
	p=new CUDABilateralFilter(r, gs, gr);

	Process(img, p, NAME);
	return 0;
}

