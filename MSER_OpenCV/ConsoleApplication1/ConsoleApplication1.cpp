//for showing image loaded using iplimage we use cvshowimage which takes too much of time
//for saving image we use cvsaveimage-saves immediately
//cvReleaseImage( &img ); // free manually allocated resource

#include <opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;


int main(int, char** argv)
{
  IplImage* img = cvLoadImage("2.PNG");
  /*
  ///converting IplImage to cv::Mat 
Mat image=cvarrToMat(img); 
  */
  IplImage* img1=cvCreateImage(cvSize(img->width,img->height),img->depth, 1 );
  cvConvertImage(img, img1,0);
IplImage* img2=cvCreateImage(cvSize(img->width,img->height),img->depth, 1 );
IplImage* img3=cvCreateImage(cvSize(img->width,img->height),img->depth, 1 );
cvSetZero(img2);cvSetZero(img3);
CvMemStorage *mem;
mem = cvCreateMemStorage(0);
CvSeq *contours = 0;
 CvSeq *ptr,*polygon;
cvMorphologyEx(img1,img1,img2,cvCreateStructuringElementEx(21,3,10,2,CV_SHAPE_RECT,NULL),CV_MOP_TOPHAT,1);
 cvSaveImage("1after cvmorphologyEx.png",img1,0);
 cvShowImage("1after cvmorphologyEx",img1);
 cvThreshold(img1,img1,70,255,CV_THRESH_BINARY);
 cvSaveImage("2thresh.png",img1,0);
 cvShowImage("2thresh",img1);
cvSetZero(img2);cvSmooth(img1, img1, CV_GAUSSIAN, 3, 3 );
 cvSaveImage("3smooth-gaussian.png",img1,0);
 cvShowImage("3smooth-gaussian",img1);
 //THE BELOW 3 LINES CAN BE COMMENTED. IT IS COMMENTED AS AFTER DILATING THEY ARE
// BECOMING SO BIG THAT THEY ARE MERGING AND FORM BIG BLOBS WHICH GET ELIMINATED AFTER AREA FILTERING
//cvDilate(img1,img1,cvCreateStructuringElementEx(100,3,50,2,CV_SHAPE_RECT,NULL),1);
// cvSaveImage("4Dilated image.png",img1,0);
// cvShowImage("4Dilated image",img1);
cvCanny(img1,img1,30,90,3);//CAN CHANGE 30 AND 90
cvSaveImage("5canny-edged image.png",img1,0);
cvShowImage("5canny-edged image",img1);
cvFindContours(img1, mem, &contours, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
for (ptr =contours; ptr != NULL; ptr = ptr->h_next) 
 	{
		double reg = fabs(cvContourArea(ptr, CV_WHOLE_SEQ));

			if(reg >0 && reg <5000)//CAN CHANGE

			{

  			CvScalar ext_color = CV_RGB( 255, 255, 255 ); //randomly coloring different contours
			cvDrawContours(img3, ptr,ext_color,CV_RGB(0,0,0), -1, CV_FILLED, 8, cvPoint(0,0));
			CvRect rectEst = cvBoundingRect( ptr, 0 );
			 CvPoint pt1,pt2;
                 pt1.x = rectEst.x; pt1.y = rectEst.y;
                pt2.x = rectEst.x+ rectEst.width; pt2.y = rectEst.y+ rectEst.height;
				int thickness =2 ;
                 cvRectangle( img, pt1, pt2, CV_RGB(0,255,0), thickness );
                 cvRectangle( img3, pt1, pt2, CV_RGB(0,255,0 ), thickness );
 			//display( img);
			cvSetImageROI(img,rectEst);
			cvSaveImage("6after setting image ROI.png",img,0);
			cvShowImage("6after setting image ROI",img);
			cvResetImageROI(img);
			}
	}
cvSaveImage("7Detection-normal.png",img,0);
	  cvSaveImage("8blobs.png",img3,0);
	  cin.get();
    return 0;
}
