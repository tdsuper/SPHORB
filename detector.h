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

#ifndef _DETECTOR_H
#define _DETECTOR_H

#include <opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

typedef CvPoint xy;																			
typedef unsigned char byte;																	

xy* sfast_corner_detect(const byte* im, const byte* mask, int xsize, int xstride, int ysize, int barrier, int* num);

int sfast_corner_score(const byte* im, const int pixel[], int bstart);

int* sfastScore(const unsigned char* i, int stride, xy* corners, int num_corners, int b);

void sfastNonmaxSuppression(const xy* corners, const int* scores, int num_corners, vector<KeyPoint>& kps, int partIndex);

#endif