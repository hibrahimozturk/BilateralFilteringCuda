#ifndef TIMER_H 
#define TIMER_H 

#include <iostream>

using std::cout;
using std::endl;

class Timer{
public:
	Timer();
	void start();
	void stop();
	void printTime();
	float getTime();

private:
	cudaEvent_t begin, end;
};

#endif

