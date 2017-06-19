/*
	AUTHOR:
	Qiang Zhao, email: qiangzhao@tju.edu.cn
	Copyright (C) 2015 Tianjin University
	School of Computer Software
	School of Computer Science and Technology

	LICENSE:
	SPHORB is distributed under the GNU General Public License.  For information on 
	commercial licensing, please contact the authors at the contact address below.

	REFERENCE:
	@article{zhao-SPHORB,
	author   = {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
	title    = {SPHORB: A Fast and Robust Binary Feature on the Sphere},
	journal  = {International Journal of Computer Vision},
	year     = {2015},
	volume   = {113},
	number   = {2},
	pages    = {143-159},
	}
*/

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SPHORB.h"
#include "utility.h"
using namespace std;
using namespace cv;

int main(int argc, char * argv[])
{
	float ratio = 0.75f;
	SPHORB sorb;

	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);

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

	imwrite("1_matches.jpg", imgMatches);

	return 0;
}