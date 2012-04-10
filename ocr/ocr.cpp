/************************/
/* Includes				*/
/************************/
#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include "basicOCR.h"

using namespace std;

/************************/
/* Declared Vars		*/
/************************/
// Image objects
IplImage* imageN;
IplImage* smoothImg;
IplImage* thresholdImg;
IplImage* imgContour;
IplImage* cropped;

// Contour Objects
CvSeq* contour;
CvSeq* contourLow;

// Color Object
CvScalar color = CV_RGB( rand()&200, rand()&200, rand()&200 );

// Ocr Object
basicOCR ocr;

// Window title
const char *ocrWindow = "Number Detection";


/************************/
/* Functions			*/
/************************/
// Crop Image to find 1 number
IplImage* CropAndScale( IplImage* src,  CvRect roi){
  // Must have dimensions of output image
  IplImage* cropped = cvCreateImage( cvSize(roi.width,roi.height), src->depth, src->nChannels );
  IplImage* scaled = cvCreateImage( cvSize(50,50), src->depth, src->nChannels );

  // Say what the source region is
  cvSetImageROI( src, roi );

  // Do the copy
  cvCopy( src, cropped );
  cvResetImageROI( src );

  // Scale it to 50x50 fixed
  cvResize(cropped,scaled,1);

  return scaled;
}


/************************/
/* Main					*/
/************************/
int _tmain(int argc, _TCHAR* argv[])
{
	srand ( time(NULL) );
	
	//Load captcha img
	char* filename= "c/2.png"; // argv[1]?

	//Create window
	cvNamedWindow( ocrWindow, 0 );

	//load image in greyscale
	imageN = cvLoadImage(filename, 0);

	//Create needed images

	smoothImg = cvCreateImage(cvSize(imageN->width, imageN->height), IPL_DEPTH_8U, 1);
	thresholdImg = cvCreateImage(cvSize(imageN->width, imageN->height), IPL_DEPTH_8U, 1);

	//Smooth image
	cvSmooth(imageN, smoothImg, CV_MEDIAN, 3, 0, 0, 0);
	
	CvScalar avg;
	CvScalar avgStd;
	cvAvgSdv(smoothImg, &avg, &avgStd, NULL);

	cout << "Avg: " << avg.val[0] << "\nStd: " << avgStd.val[0] <<"\n";

	//threshold image
	cvThreshold(smoothImg, thresholdImg, (int)avg.val[0]-7*(int)(avgStd.val[0]/8), 255, CV_THRESH_BINARY_INV);

	//Init variables for countours
	contour = 0;
	contourLow = 0;

	//Create storage needed for contour detection
	CvMemStorage* storage = cvCreateMemStorage(0);

	//Duplicate image for countour
	imgContour = cvCloneImage(thresholdImg);

	//Search countours in preprocesed image
	cvFindContours( imgContour, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, cvPoint(0, 0) );
	
	//Optimize contours, reduce points
	contourLow=cvApproxPoly(contour, sizeof(CvContour), storage,CV_POLY_APPROX_DP, 1, 1);
	
	//For each contour found
	for( ; contourLow != 0; contourLow = contourLow->h_next )
	{
		//Detect bounding rect of number	
		CvRect rect = cvBoundingRect(contourLow, NULL);
		
		// If the contour is small as hell, discard (noise)
		if (rect.width < 22) 
			continue;
		if (rect.height < 25)
			continue;

		CvPoint pt1, pt2;
		pt1.x = rect.x;
        pt2.x = (rect.x+rect.width);
        pt1.y = rect.y;
        pt2.y = (rect.y+rect.height);
		
		cout << "width: " << rect.width << ", height:" << rect.height << ".\n";

		// Sleep to give it time and draw the contour on the number
		Sleep(5);
		cvRectangle(thresholdImg, pt1,pt2, color, 1, 8, 0); 
		Sleep(5);
		
		// Save each number as its own image
		// Saves the number images in order of occurance
		cropped = CropAndScale(thresholdImg, rect);

		if (rect.x <= 45) {
			ocr.classify(cropped,1);
			cvSaveImage("1.jpg", cropped);
		}
		if (rect.x > 45 && rect.x <= 90){
			ocr.classify(cropped,2);
			cvSaveImage("2.jpg", cropped);
		}
		if (rect.x > 90 && rect.x <= 140){
			ocr.classify(cropped,3);
			cvSaveImage("3.jpg", cropped);
		}
		if (rect.x > 140){
			ocr.classify(cropped,4);
			cvSaveImage("4.jpg", cropped);
		}
	}
	
	cvShowImage(ocrWindow, thresholdImg);
	cvResizeWindow(ocrWindow, 550, 550);
	

	//forever a loop
	// Try pressing these keys
	for(;;)
	{
		int c = cvWaitKey(10);

		if( (char) c == 27 )
			break;
		else if((char) c=='1')
			cvShowImage(ocrWindow, imageN);
		else if((char) c=='2')
			cvShowImage(ocrWindow, smoothImg);
		else if((char) c=='3')
			cvShowImage(ocrWindow, thresholdImg);
	}

	//Being nice 2 my computard
	cvDestroyWindow(ocrWindow);
	return 0;
}

