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

#ifndef _SPHORB_H
#define _SPHORB_H

#include <opencv.hpp>		
#include <vector>
using namespace cv;

namespace cv
{
	class CV_EXPORTS SPHORB : public cv::Feature2D
	{
	public:
		enum { kBytes = 32, SFAST_EDGE = 3, SPHORB_EDGE = 15};

		explicit SPHORB(int nfeatures = 500, int nlevels = 7, int b=20);
		~SPHORB();

		// returns the descriptor size in bytes
		int descriptorSize() const;
		// returns the descriptor type
		int descriptorType() const;

		// Compute the ORB features and descriptors on an image
		void operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const;
		void operator()( InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false ) const;

	protected:
		int barrier;
		int nfeatures;
		int nlevels;

		void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
		void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;
	};

}

#endif