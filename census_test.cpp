#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <malloc.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>

#include <opencv2/core/core.hpp>  
#include <opencv2/features2d/features2d.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/calib3d/calib3d.hpp>  
#include <opencv.hpp>  

#include "CensusMatch.h"
#include "gdal_priv.h"
//#include "reconstruct3D_11_5.h"

using namespace cv;

bool writeImage(const char* filename, float32 * dispData, int width, int height)
{

	if (height <= 0 || width <= 0 || dispData == NULL)
	{
		std::cout << "create image failed!" << std::endl;
		return false;
	}

	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");

	GDALDriver * driver = GetGDALDriverManager()->GetDriverByName("GTiff");
	if (driver == NULL)
	{
		std::cout << "format not support!" << std::endl;
		return false;
	}

	GDALDataset * dataset = driver->Create(filename, width, height, 1, GDT_Float32, NULL);
	dataset->RasterIO(GF_Write, 0, 0, width, height, dispData, width, height, GDT_Float32, 1, 0, 0, 0, 0, NULL);

	GDALClose(dataset);
}

int main()
{
	// read
	const char* leftFileName = "(2)L.png";
	const char* rightFileName = "(2)R.png";

	Mat imgOriginal1 = imread(leftFileName, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgOriginal2 = imread(rightFileName, CV_LOAD_IMAGE_GRAYSCALE);

	Mat img1, img2;
	img1 = imgOriginal1;
	img2 = imgOriginal2;
	//double ratio = imgOriginal1.cols / 16;
	//int cols = int(ratio) * 16; 
	//double ddddddd = double(cols) / imgOriginal1.cols;

	//resize(imgOriginal1, img1, Size(imgOriginal1.cols * ddddddd, imgOriginal1.rows * ddddddd));
	//resize(imgOriginal2, img2, Size(imgOriginal2.cols * ddddddd, imgOriginal2.rows * ddddddd));
	//
	//Mat textureSave1, textureSave2;
	//Mat texture1 = imread(leftFileName, CV_LOAD_IMAGE_UNCHANGED);
	//Mat texture2 = imread(rightFileName, CV_LOAD_IMAGE_UNCHANGED);

	//resize(texture1, textureSave1, Size(texture1.cols * ddddddd, texture1.rows * ddddddd));
	//resize(texture2, textureSave2, Size(texture2.cols * ddddddd, texture2.rows * ddddddd));
	//imwrite("left.tif", textureSave1);
	//imwrite("right.tif", textureSave2);
	

	clock_t begin = clock();
	// compute
	CensusMatch match;

	float32* dispImgLeft = (float32*)_mm_malloc(img1.cols * img1.rows * sizeof(float32), 16);
	float32* dispImgRight = (float32*)_mm_malloc(img2.cols * img2.rows * sizeof(float32), 16);

	match.processCensus5x5(img1.data, img2.data, dispImgLeft, dispImgRight, img1.cols, img1.rows, 128);

	// save 
	float32* dispImgSaveLeft = (float32*)_mm_malloc(img1.cols * img1.rows * sizeof(float32), 16);
	float32* dispImgSaveRight = (float32*)_mm_malloc(img2.cols * img2.rows * sizeof(float32), 16);

	//match.medianNxN(dispImg, dispImgSave, img1.cols, img1.rows, 3);
	//match.medianNxN(dispImgRight, dispImgRightSave, img1.cols, img1.rows, 3);
	match.median3x3_SSE(dispImgLeft, dispImgSaveLeft, img1.cols, img1.rows);
	match.median3x3_SSE(dispImgRight, dispImgSaveRight, img1.cols, img1.rows);

	match.doLRCheck(dispImgSaveLeft, dispImgSaveRight, img1.cols, img1.rows, 1);



	clock_t end = clock();
	double elapsed_secs = double(end - begin) * 1000 / CLOCKS_PER_SEC;
	printf("TIME£º %.2lf ms.\n", elapsed_secs);

	// save disparity image
	bool re = writeImage("disparity_census_leftM.tif", dispImgSaveLeft, img1.cols, img1.rows);
	bool re2 = writeImage("disparity_census_rightM.tif", dispImgSaveRight, img1.cols, img1.rows);

	//int stage = reconstrc3D("left.tif", "right.tif", "", "", "myTest//disparity_census.tif", "myTest//points.txt");
	//int stage = reconstrc3D("left.tif", "right.tif", "", "", "myTest//first_Filter.tif", "myTest//pointsSGM_desktop.txt");

	return 1;
}