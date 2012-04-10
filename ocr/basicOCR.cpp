#ifndef _EiC

#include <iostream>
#include "cv.h"
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#endif
#include "StdAfx.h"
#include "preprocessing.h"
#include "basicOCR.h"
using namespace std;



basicOCR::basicOCR() {
	//initial
	char file_path[] = "C:\Users\TheSisb\Documents\Visual Studio 2010\Projects\ocr\c\OCR";
	cout << file_path;
	train_samples = 108;
	classes= 9;
	size=50; //50?

	trainData = cvCreateMat(train_samples*classes, size*size, CV_32FC1);
	trainClasses = cvCreateMat(train_samples*classes, 1, CV_32FC1);

	//Get data (get images and process it)
	getData();
	
	//train	
	train();
	
	printf(" ---------------------------------------------------------------\n");
	printf("|\tClass\t|\tPrecision\t|\tAccuracy\t|\n");
	printf(" ---------------------------------------------------------------\n");
}





void basicOCR::getData()
{
	char file_path[] = "C:/Users/TheSisb/Documents/Visual Studio 2020/Projects/opencv1/opencv3/OCR";
	IplImage* src_image;
	IplImage prs_image;
	CvMat row, data;
	char file[255];
	int i,j;

	for(i = 0; i < classes; i++){
		for( j = 0; j < train_samples; j++){
					
			sprintf(file, "%s/%d/%d.jpg",file_path, i+1, j+1);
			
			src_image = cvLoadImage(file,0);

			if(!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			

			//Set class label
			cvGetRow(trainClasses, &row, i*train_samples + j);
			cvSet(&row, cvRealScalar(i));

			//Set data 
			cvGetRow(trainData, &row, i*train_samples + j);

			//Create new container image
			IplImage* img = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );

			//convert 8 bits image to 32 float image
			cvConvertScale(&prs_image, img, 0.0039215, 0);
			cvGetSubRect(img, &data, cvRect(0,0, size,size));
			CvMat row_header, *row1;

			//convert data matrix sizexsize to vecor
			row1 = cvReshape( &data, &row_header, 0, 1 );
			cvCopy(row1, &row, NULL);
		}
	}
}

void basicOCR::train() {
	knn = new CvKNearest( trainData, trainClasses, 0, false, K );
}

float basicOCR::classify(IplImage* img, int showResult)
{
	IplImage prs_image;
	CvMat data;
	CvMat* nearest = cvCreateMat(1, K, CV_32FC1);
	float result;

	//process file
	prs_image = preprocessing(img, size, size);
	

	//Set data 
	IplImage* img32 = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
	cvConvertScale(&prs_image, img32, 0.0039215, 0);
	cvGetSubRect(img32, &data, cvRect(0,0, size,size));
	CvMat row_header, *row1;
	row1 = cvReshape( &data, &row_header, 0, 1 );

	result = knn->find_nearest(row1, K, 0, 0, nearest, 0);
	
	int accuracy=0;

	for(int i=0; i<K; i++){
		if(nearest->data.fl[i] == result)
			accuracy++;
	}

	float pre = 100*((float)accuracy/(float)K);

	if(showResult > 0 &&showResult <= 4){
		printf("|\t%.0f\t| \t%.2f%%  \t| \t%d of %d \t| %d \n",result+1,pre,accuracy,K, showResult);
		printf(" ---------------------------------------------------------------\n");
	}

	return result;

}