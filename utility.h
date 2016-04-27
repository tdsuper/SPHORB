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

#ifndef _UTILITY_H
#define _UTILITY_H

#include <vector>
#include <opencv.hpp>
using namespace cv;

typedef vector<DMatch> Matches;
void ratioTest(const std::vector<Matches>& knMatches, float maxRatio, Matches& goodMatches);

void drawMatches(const Mat& img1, const vector<KeyPoint>& keypoints1,
	const Mat& img2, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& matches1to2, Mat& outImg, const Scalar& matchColor, const Scalar& singlePointColor,
	const vector<char>& matchesMask, int flags , bool vertical);

#endif