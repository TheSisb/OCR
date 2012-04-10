#ifndef _EiC
#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <stdio.h>
#include <ctype.h>
#endif

class basicOCR{

	public:
		basicOCR ();
		float classify(IplImage* img,int showResult);
		void test();

	private:
		//vars
		static const int K=8;
		char file_path[255];
		int train_samples;
		int classes;
		int size;
		CvKNearest *knn;
		CvMat* trainData;
		CvMat* trainClasses;

		//functions
		void getData();
		void train();
};
