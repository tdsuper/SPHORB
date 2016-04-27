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

#include "utility.h"

// ratio matching strategy
void ratioTest(const std::vector<Matches>& knMatches, float maxRatio, Matches& goodMatches)
{
	goodMatches.clear();

	for (size_t i=0; i< knMatches.size(); i++)
	{
		const cv::DMatch& best = knMatches[i][0];
		const cv::DMatch& good = knMatches[i][1];

		assert(best.distance <= good.distance);
		float ratio = (best.distance / good.distance);

		if (ratio <= maxRatio)
		{
			goodMatches.push_back(best);
		}
	}
}


// The following code is used to draw the matches, so that the first image is on top of the second image.
// Note that the first image is at the left of the second image, if you use the "drawMatches" function of OpenCV.
const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

static inline void _drawKeypoint( Mat& img, const KeyPoint& p, const Scalar& color, int flags )
{
	CV_Assert( !img.empty() );
	Point center( cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );

	if( flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS )
	{
		int radius = cvRound(p.size/2 * draw_multiplier); // KeyPoint::size is a diameter

		// draw the circles around keypoints with the keypoints size
		circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );

		// draw orientation of the keypoint, if it is applicable
		if( p.angle != -1 )
		{
			float srcAngleRad = p.angle*(float)CV_PI/180.f;
			Point orient( cvRound(cos(srcAngleRad)*radius ),
				cvRound(sin(srcAngleRad)*radius )
				);
			line( img, center, center+orient, color, 1, CV_AA, draw_shift_bits );
		}
#if 0
		else
		{
			// draw center with R=1
			int radius = 1 * draw_multiplier;
			circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
		}
#endif
	}
	else
	{
		// draw center with R=3
		int radius = 3 * draw_multiplier;
		circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
	}
}

static void _prepareImgAndDrawKeypoints( const Mat& img1, const vector<KeyPoint>& keypoints1,
	const Mat& img2, const vector<KeyPoint>& keypoints2,
	Mat& outImg, Mat& outImg1, Mat& outImg2,
	const Scalar& singlePointColor, int flags )
{
	cv::Size size( MAX(img1.cols, img2.cols), img1.rows + img2.rows);
	if( flags & DrawMatchesFlags::DRAW_OVER_OUTIMG )
	{
		if( size.width > outImg.cols || size.height > outImg.rows )
			CV_Error( CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
		outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
		outImg2 = outImg( Rect(0, img1.rows, img2.cols, img2.rows) );
	}
	else
	{
		outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
		outImg = Scalar::all(0);
		outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
		outImg2 = outImg( Rect(0, img1.rows, img2.cols, img2.rows) );

		if( img1.type() == CV_8U )
			cvtColor( img1, outImg1, CV_GRAY2BGR );
		else
			img1.copyTo( outImg1 );

		if( img2.type() == CV_8U )
			cvtColor( img2, outImg2, CV_GRAY2BGR );
		else
			img2.copyTo( outImg2 );
	}

	// draw keypoints
	if( !(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) )
	{
		Mat _outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
		drawKeypoints( _outImg1, keypoints1, _outImg1, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );

		Mat _outImg2 = outImg( Rect(0, img1.rows, img2.cols, img2.rows) );
		drawKeypoints( _outImg2, keypoints2, _outImg2, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG );
	}
}

static inline void _drawMatch( Mat& outImg, Mat& outImg1, Mat& outImg2 ,
	const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, int flags )
{
	RNG& rng = theRNG();
	bool isRandMatchColor = matchColor == Scalar::all(-1);
	Scalar color = isRandMatchColor ? Scalar( rng(256), rng(256), rng(256) ) : matchColor;

	_drawKeypoint( outImg1, kp1, color, flags );
	_drawKeypoint( outImg2, kp2, color, flags );

	Point2f pt1 = kp1.pt,
		pt2 = kp2.pt,
		dpt2 = Point2f(pt2.x, std::min(pt2.y+outImg1.rows, float(outImg.rows-1)));

	line( outImg,
		Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
		Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
		color, 1, CV_AA, draw_shift_bits );
}

void drawMatches(const Mat& img1, const vector<KeyPoint>& keypoints1,
	const Mat& img2, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& matches1to2, Mat& outImg, const Scalar& matchColor, const Scalar& singlePointColor,
	const vector<char>& matchesMask, int flags , bool vertical)
{
	if(!vertical)
		cv::drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags);
	else
	{
		if( !matchesMask.empty() && matchesMask.size() != matches1to2.size() )
			CV_Error( CV_StsBadSize, "matchesMask must have the same size as matches1to2" );

		Mat outImg1, outImg2;
		_prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2,
			outImg, outImg1, outImg2, singlePointColor, flags );

		// draw matches
		for( size_t m = 0; m < matches1to2.size(); m++ )
		{
			int i1 = matches1to2[m].queryIdx;
			int i2 = matches1to2[m].trainIdx;
			if( matchesMask.empty() || matchesMask[m] )
			{
				const KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
				_drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor, flags );
			}
		}
	}
}