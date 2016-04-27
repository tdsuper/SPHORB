/*
	AUTHOR:
	Qiang Zhao, email: qiangzhao@tju.edu.cn
	Copyright (C) 2014 Tianjin University
	School of Computer Software
	School of Computer Science and Technology

	LICENSE:
	SPHORB is distributed under the GNU General Public License.  For information on 
	commercial licensing, please contact the authors at the contact address below.

	REFERENCE:
	@TECHREPORT{SPHORB2014,
	author =       {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
	title =        {SPHORB: A Fast and Robust Binary Feature on the Sphere},
	institution =  {Tianjin University},
	year =         {2014},
	}
*/

#include <iostream>
#include <vector>
#include <opencv.hpp>
#include "SPHORB.h"
#include "utility.h"
using namespace std;
using namespace cv;

int main()
{
	float ratio = 0.75f;
	SPHORB sorb(10000);

	Mat img1 = imread("Image/2_1.jpg");
	Mat img2 = imread("Image/2_2.jpg");

	Mat descriptors1;
	Mat descriptors2;

	vector<KeyPoint> kPoint1;
	vector<KeyPoint> kPoint2;

	sorb(img1, Mat(), kPoint1, descriptors1);
	sorb(img2, Mat(), kPoint2, descriptors2); 

	cout<<"Keypoint1: "<<kPoint1.size()<<", Keypoint2: "<<kPoint2.size()<<endl;

	BFMatcher matcher(NORM_HAMMING, false);
	Matches matches;
	
	vector<Matches> dupMatches;
	matcher.knnMatch(descriptors1, descriptors2, dupMatches, 2);
	ratioTest(dupMatches, ratio, matches);
	cout<<"Matches: "<<matches.size()<<endl;

	Mat imgMatches;
	::drawMatches(img1, kPoint1, img2, kPoint2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1),  
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,true);

	imwrite("2_matches.jpg", imgMatches);

	return 0;
}