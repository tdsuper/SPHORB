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

#include <algorithm>
#include <iterator>
#include "SPHORB.h"
#include "pfm.h"
#include "detector.h"

#define MAX_PATH 15

namespace cv
{
	std::vector<float*> geoinfos;
	std::vector<Mat> maskes;
	std::vector<vector<float*> > imgInfos;

	const int cells[] = {256, 204, 162, 128, 102, 80, 64};
	int levels;

	// load the precomputed information
	static void initSORB()
	{
		levels = sizeof(cells) / sizeof(cells[0]);
		for (int i=0;i<levels;i++)
		{
			// geodesic grid coordinate with different resolution
			char fileName[MAX_PATH];
			sprintf(fileName, "Data/geoinfo%d.pfm", cells[i]);

			float* geoinfo = new float[(cells[i]+1)*(2*cells[i]+1)*3];
			read_pfm(fileName, geoinfo);
			geoinfos.push_back(geoinfo);

			// look up table for fast image convertion from spherical image to geodesic grid
			vector<float*> partInfos;
			for (int j=0;j<5;j++)
			{
				sprintf(fileName, "Data/imginfo%d_%d.pfm", cells[i], j);

				int num = (2*cells[i]+1)*(cells[i]+1)*4;
				num = static_cast<int>(ceil(num / 3.0)) * 3;
				float* partinfo = new float[num];
				read_pfm(fileName, partinfo);
				partInfos.push_back(partinfo);
			}
			imgInfos.push_back(partInfos);

			// the mask image
			sprintf(fileName, "Data/mask%d.bmp", cells[i]);
			Mat mask = imread(fileName, 0);
			maskes.push_back(mask);
		}
	}

	static void uninitSORB()
	{
		for (size_t i=0;i<geoinfos.size();i++)
		{
			if (geoinfos[i]!=NULL)
			{
				delete[] geoinfos[i];
				geoinfos[i] = NULL;
			}
		}
		for (size_t i=0;i<imgInfos.size();i++)
		{
			for (size_t j=0;j<imgInfos[i].size();j++)
			{
				if (imgInfos[i][j]!=NULL)
				{
					delete [] imgInfos[i][j];
					imgInfos[i][j] = NULL;
				}
			}
		}

		geoinfos.clear();
		imgInfos.clear();
		maskes.clear();
	}

// split spherical image to the storage grid
static void splitSphere2(const Mat& im, Mat& oim, int idx, const float* imgInfo)
{
	for(int y=0; y<oim.rows; y++)
	{
		for(int x=0; x<oim.cols; x++)
		{
			float lx = imgInfo[(x+y*oim.cols)*4];
			float ly = imgInfo[(x+y*oim.cols)*4+1];
			float wh = imgInfo[(x+y*oim.cols)*4+2];
			float wv = imgInfo[(x+y*oim.cols)*4+3];

			int ix = static_cast<int>(lx);
			int iy = static_cast<int>(ly);

			uchar v1 = im.at<uchar>(iy, ix);
			uchar v2 = im.at<uchar>(iy, (ix+1)%im.cols);
			uchar v3 = im.at<uchar>(iy+1, ix);
			uchar v4 = im.at<uchar>(iy+1, (ix+1)%im.cols);

			float v12 = v1*wh + v2*(1-wh);
			float v34 = v3*wh + v4*(1-wh);
			oim.at<uchar>(y, x) = uchar(v12*wv + v34*(1-wv));
		}
	}

}

// extend the storage grid from top and right boundary
static void extendTopRight(Mat& newPart1, const Mat& part2, int edge)
{
	int h = part2.rows;
	int w = part2.cols;

	int r, c;

	r = edge;
	for(c=edge-1; c<h+edge-1; c++)
	{

		int c0 = c-edge+1;
		int rn = c0;
		int cn = 0;
		for(int i=1;i<=edge;i++)
		{
			rn--;
			cn++;
			if(rn>=0)
				newPart1.at<uchar>(r-i, c) = part2.at<uchar>(rn, cn);
			else
				break;
		}
	}

	for(c=h+edge-1; c<w+edge-1; c++)
	{
		int c0 = c-edge+1;
		int rn=h-1;
		int cn = c0-h+1;
		for(int i=1;i<=edge;i++)
		{
			rn--;
			if(rn+cn>=h-1)
				newPart1.at<uchar>(r-i, c) = part2.at<uchar>(rn, cn);
			else
				break;
		}
	}

	c = w+edge-2;
	for(r=edge;r<h+edge;r++)
	{
		int r0 = r-edge;
		int rn = h-1;
		int cn = r0+h-1;
		for(int i=1;i<=edge;i++)
		{
			rn--;
			cn++;
			if(cn<2*h-1)
				newPart1.at<uchar>(r, c+i) = part2.at<uchar>(rn, cn);
			else
				break;

		}
	}
}

// extend the storage grid from left and bottom boundary
static void extendBottomLeft(Mat& newPart1, const Mat& part2, int edge)
{
	int h = part2.rows;
	int w = part2.cols;

	int r, c;

	c = edge-1;
	for(r=edge; r<h+edge; r++)
	{
		int r0 = r-edge;
		int c0 = c-edge+1;
		int rn = c0;
		int cn = r0;
		for(int i=1;i<=edge-1;i++)
		{
			rn++;
			cn--;
			if(cn>=0)
				newPart1.at<uchar>(r, c-i) = part2.at<uchar>(rn, cn);
			else
				break;
		}
	}

	r = h+edge-1;
	for(c=edge-1; c<h+edge-2; c++)
	{
		int c0 = c-edge+1;
		int rn = 0;
		int cn = c0+h-1;
		for(int i=1;i<=edge-1;i++)
		{
			rn++;
			if(rn+cn<=2*h-2)
				newPart1.at<uchar>(r+i, c) = part2.at<uchar>(rn, cn);
			else
				break;
		}
	}


	for(c=h+edge-2;c<w+edge-1;c++)
	{
		int c0 = c-edge+1;
		int rn = c0-h+1;
		int cn = 2*h-2;
		for(int i=1;i<=edge-1;i++)
		{
			rn++;
			cn--;
			if(rn<h)
				newPart1.at<uchar>(r+i, c) = part2.at<uchar>(rn, cn);
			else
				break;

		}
	}
}

// extend the storage grid
static void extendEdge(Mat& part0, Mat& part1, Mat& part2, Mat& part3, Mat& part4, int edge)
{
	int height = part0.rows + edge*2 - 1;
	int width = part0.cols + edge*2 - 1;

	// extend the image size
	Mat _part0 = Mat::zeros(Size(width, height), part0.type());
	Mat _part1 = Mat::zeros(Size(width, height), part1.type());
	Mat _part2 = Mat::zeros(Size(width, height), part2.type());
	Mat _part3 = Mat::zeros(Size(width, height), part3.type());
	Mat _part4 = Mat::zeros(Size(width, height), part4.type());

	// copy the source data
	for (int y=edge; y<height-edge+1; y++)
	{
		for (int x=edge-1; x<width-edge; x++)
		{
			_part0.at<uchar>(y,x) = part0.at<uchar>(y-edge, x-edge+1);
			_part1.at<uchar>(y,x) = part1.at<uchar>(y-edge, x-edge+1);
			_part2.at<uchar>(y,x) = part2.at<uchar>(y-edge, x-edge+1);
			_part3.at<uchar>(y,x) = part3.at<uchar>(y-edge, x-edge+1);
			_part4.at<uchar>(y,x) = part4.at<uchar>(y-edge, x-edge+1);

		}
	}

	// extend the edges
	extendTopRight(_part0, part1, edge);
	extendTopRight(_part1, part2, edge);
	extendTopRight(_part2, part3, edge);
	extendTopRight(_part3, part4, edge);
	extendTopRight(_part4, part0, edge);

	extendBottomLeft(_part1, part0, edge);
	extendBottomLeft(_part2, part1, edge);
	extendBottomLeft(_part3, part2, edge);
	extendBottomLeft(_part4, part3, edge);
	extendBottomLeft(_part0, part4, edge);

	// copy the extend image
	part0 = _part0;
	part1 = _part1;
	part2 = _part2;
	part3 = _part3;
	part4 = _part4;
}

// the angle between the x-axis of local coordinate and the south pole
static float inherentAngle(const float* center, const float* axisx)
{
	float t1 = -center[0]*axisx[0]-center[1]*axisx[1]-center[2]*axisx[2]+1.0f;
	float x1 =  axisx[0] + center[0] * t1;
	float y1 =  axisx[1] + center[1] * t1;
	float z1 =  axisx[2] + center[2] * t1;

	float t2 = center[2]+1.0f;
	float x2 = center[0] * t2;
	float y2 = center[1] * t2;
	float z2 = -1 + center[2] * t2;

	x1 -= center[0];
	y1 -= center[1];
	z1 -= center[2];

	x2 -= center[0];
	y2 -= center[1];
	z2 -= center[2];

	float tmp = (x1*x2+y1*y2+z1*z2) / ( sqrt( (x1*x1+y1*y1+z1*z1) * (x2*x2+y2*y2+z2*z2) ) );
	if (tmp>1.0f)
		tmp = 1.0f;
	if(tmp<-1.0f)
		tmp = -1.0f;
	float radian = acos( tmp );

	if (center[0]*axisx[1]-axisx[0]*center[1]<0)
		return  (float)(-radian*(180.f/CV_PI));
	else
		return (float)(radian*(180.f/CV_PI));
}

static float IC_Angle(const Mat& img, const int half_k, Point2f pt, const float* geoinfo)
{
	float m_01 = 0, m_10 = 0;

	int step = (int)img.step1();
	const uchar* center = &img.at<uchar> (cvRound(pt.y), cvRound(pt.x));

	float tmp = sqrt(3.0f) * 0.5f;

	for(int y=-half_k; y<=half_k; y++)
	{
		int xmax, xmin;
		if(y<0)
		{
			xmin = -y-half_k;
			xmax = half_k;
		}
		else
		{
			xmin = -half_k;
			xmax = half_k-y;
		}

		for(int x=xmin; x<=xmax; x++)
		{
			float euclidX = x + y * 0.5f;
			float euclidY = tmp * y;

			m_10 += euclidX * center[x+y*step];
			m_01 += euclidY * center[x+y*step];

		}

	}

	return fastAtan2((float)m_01, (float)m_10);

	//int x = pt.x - half_k - 2;
	//int y = pt.y - half_k - 3;
	//return fastAtan2((float)m_01, (float)m_10) - inherentAngle(geoinfo+(x+y*513)*3, geoinfo+(x+1+y*513)*3);
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& kps, int halfPatchSize, const float* geoinfo)
{
	for(size_t i=0;i<kps.size();i++)
	{
		kps[i].angle = IC_Angle(image, halfPatchSize, kps[i].pt, geoinfo);
	}
}

static void computeOrbDescriptor(const KeyPoint& kpt, const Mat& img, const Point* pattern, byte* desc, int dsize)
{
	float angle = kpt.angle;
	angle *= (float)(CV_PI/180.f);
	float a = (float)cos(angle), b = (float)sin(angle);
	float c = sqrt(3.0f);
	float d = b*c/3;
	b = a - d;
	a = a + d;
	c = 2 * d;

	int step = (int)img.step;
	const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));

	// Transform the sampling patterns from rectangle structure to the Euclidean space, 
	// after rotating the patterns, transform them to the original space.
#define GET_VALUE(idx) \
	center[cvRound(pattern[idx].y*a + pattern[idx].x*c)*step + \
	cvRound(pattern[idx].x*b - pattern[idx].y*c)]

	for (int i = 0; i < dsize; ++i, pattern += 16)
	{
		int t0, t1, val;
		t0 = GET_VALUE(0); t1 = GET_VALUE(1);
		val = t0 < t1;
		t0 = GET_VALUE(2); t1 = GET_VALUE(3);
		val |= (t0 < t1) << 1;
		t0 = GET_VALUE(4); t1 = GET_VALUE(5);
		val |= (t0 < t1) << 2;
		t0 = GET_VALUE(6); t1 = GET_VALUE(7);
		val |= (t0 < t1) << 3;
		t0 = GET_VALUE(8); t1 = GET_VALUE(9);
		val |= (t0 < t1) << 4;
		t0 = GET_VALUE(10); t1 = GET_VALUE(11);
		val |= (t0 < t1) << 5;
		t0 = GET_VALUE(12); t1 = GET_VALUE(13);
		val |= (t0 < t1) << 6;
		t0 = GET_VALUE(14); t1 = GET_VALUE(15);
		val |= (t0 < t1) << 7;

		desc[i] = (uchar)val;
	}
	
#undef GET_VALUE
}

static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints , Mat& descriptors,
	const vector<Point>& pattern, int dsize, int WTA_K)
{

	descriptors = Mat::zeros(keypoints.size(), dsize, CV_8UC1);


	for (size_t i = 0; i < keypoints.size(); i++)
	{
		computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i), dsize);
	}
}
// map the keypoint of each level of the five part of the storage grid to the original spherical image
static void mappingKeypoint(const Mat& img, vector<cv::KeyPoint>& kps, int edge, const float* geoinfo, int level)
{

	float scale = float(cells[0])/float(cells[level]);
	int pWidth = cells[level]*2+1;

	float pcos[5] = {cos(0.0), cos(2*CV_PI/5), cos(4*CV_PI/5), cos(6*CV_PI/5), cos(8*CV_PI/5)};
	float psin[5] = {sin(0.0), sin(2*CV_PI/5), sin(4*CV_PI/5), sin(6*CV_PI/5), sin(8*CV_PI/5)};
	float c = CV_PI/img.rows;
	for (size_t i=0;i<kps.size();i++)
	{
		int _x = static_cast<int>(kps[i].pt.x);
		int _y = static_cast<int>(kps[i].pt.y);

		int x = static_cast<int>(kps[i].pt.x - edge + 1);
		int y = static_cast<int>(kps[i].pt.y - edge);

		float sX3D = geoinfo[(x+y*pWidth)*3];
		float sY3D = geoinfo[(x+y*pWidth)*3+1];
		float sZ3D = geoinfo[(x+y*pWidth)*3+2];

		float dX3D = pcos[kps[i].class_id]*sX3D - psin[kps[i].class_id]*sY3D;
		float dY3D = pcos[kps[i].class_id]*sY3D + psin[kps[i].class_id]*sX3D;
		float dZ3D = sZ3D;


		float theta = acos(dZ3D);
		float phi = atan2(dY3D,dX3D) + CV_PI;

		float panoX = phi / c;
		float panoY = theta / c;

		kps[i].size = 31.0f*scale;
		kps[i].pt.x = panoX*scale;
		kps[i].pt.y = panoY*scale;
		kps[i].class_id = -1;
		kps[i].octave = level;
	}
}

// the gaussian used to smooth the storage grid
static double kernel[] = { 0, 0, 0, 0.007615469730253, 0.012684563109382, 0.012684563109382, 0.007615469730253,
	0, 0, 0.012684563109382, 0.027267400652990, 0.035191124791545, 0.027267400652990, 0.012684563109382,
	0, 0.012684563109382, 0.035191124791545, 0.058615431367971, 0.058615431367971, 0.035191124791545, 0.012684563109382,
	0.007615469730253, 0.027267400652990, 0.058615431367971, 0.075648683430860, 0.058615431367971, 0.027267400652990, 0.007615469730253,
	0.012684563109382, 0.035191124791545, 0.058615431367971, 0.058615431367971, 0.035191124791545, 0.012684563109382, 0,
	0.012684563109382, 0.027267400652990, 0.035191124791545, 0.027267400652990, 0.012684563109382, 0, 0,
	0.007615469730253, 0.012684563109382, 0.012684563109382, 0.007615469730253, 0, 0, 0};

// the sampling pattern
static int bit_pattern[256*4] =
{
	0, 13, -15, 14, /*mean (0.000196695), correlation (0)*/
	-15, 1, -14, 15, /*mean (0.00235352), correlation (0.0867396)*/
	15, -12, -4, -7, /*mean (0.00343412), correlation (0.0907921)*/
	9, -15, 2, -1, /*mean (0.00463572), correlation (0.0883058)*/
	15, -1, 5, 3, /*mean (0.0126926), correlation (0.0987857)*/
	10, -8, 13, 2, /*mean (0.0138597), correlation (0.0987887)*/
	-7, 4, -15, 6, /*mean (0.0160338), correlation (0.0988597)*/
	4, 11, -5, 13, /*mean (0.0174904), correlation (0.0941022)*/
	1, -14, 4, -10, /*mean (0.0179875), correlation (0.098813)*/
	7, -4, -9, 11, /*mean (0.0201184), correlation (0.0996794)*/
	-5, 2, -1, 4, /*mean (0.0202611), correlation (0.0999593)*/
	10, -11, 15, -8, /*mean (0.021882), correlation (0.0984285)*/
	9, 2, 9, 6, /*mean (0.0244927), correlation (0.0994798)*/
	-2, 8, -4, 14, /*mean (0.027259), correlation (0.0940522)*/
	-4, -10, -5, -5, /*mean (0.0273411), correlation (0.0986831)*/
	-1, -9, -9, -6, /*mean (0.0275832), correlation (0.0981432)*/
	-12, -1, -15, 4, /*mean (0.0298567), correlation (0.0972344)*/
	15, -6, 12, -3, /*mean (0.0306563), correlation (0.0982352)*/
	-5, -5, -8, 0, /*mean (0.0332757), correlation (0.0995229)*/
	6, 1, 2, 2, /*mean (0.0345897), correlation (0.0925487)*/
	2, -8, 4, -7, /*mean (0.0381859), correlation (0.0958004)*/
	-3, 5, -3, 8, /*mean (0.0386657), correlation (0.0910819)*/
	8, 6, 6, 9, /*mean (0.0458667), correlation (0.0944299)*/
	-10, -3, -13, -2, /*mean (0.0459575), correlation (0.0948076)*/
	15, -14, 13, -12, /*mean (0.0489183), correlation (0.0961879)*/
	-13, 12, -15, 13, /*mean (0.0500119), correlation (0.0896462)*/
	5, 10, 3, 12, /*mean (0.0518014), correlation (0.0956848)*/
	-2, 14, -2, 15, /*mean (0.0703832), correlation (0.0863209)*/
	-13, 6, -14, 7, /*mean (0.0727475), correlation (0.0964942)*/
	-13, -2, -14, -1, /*mean (0.0860172), correlation (0.0972335)*/
	-1, 2, -8, 4, /*mean (0.0284995), correlation (0.128915)*/
	10, -7, 6, -5, /*mean (0.032502), correlation (0.123528)*/
	1, -3, -1, 1, /*mean (0.0335177), correlation (0.128973)*/
	2, 1, 2, 5, /*mean (0.0335869), correlation (0.128039)*/
	8, -12, 11, -11, /*mean (0.0341488), correlation (0.129793)*/
	6, -13, 3, -12, /*mean (0.0355104), correlation (0.129379)*/
	-12, 13, -10, 15, /*mean (0.0367466), correlation (0.128868)*/
	3, -14, 0, -13, /*mean (0.0413023), correlation (0.129571)*/
	-5, 0, -6, 3, /*mean (0.0419853), correlation (0.125114)*/
	-6, -2, -3, -2, /*mean (0.0509023), correlation (0.127468)*/
	0, -15, 0, -14, /*mean (0.070206), correlation (0.10119)*/
	5, 9, 5, 10, /*mean (0.0703054), correlation (0.115902)*/
	1, -5, 0, -4, /*mean (0.0784703), correlation (0.124267)*/
	-15, 3, -13, 3, /*mean (0.08163), correlation (0.11899)*/
	3, 6, 4, 6, /*mean (0.10536), correlation (0.122715)*/
	-9, -1, -8, -1, /*mean (0.11161), correlation (0.117768)*/
	4, -3, 5, -3, /*mean (0.11652), correlation (0.128808)*/
	-11, 9, -10, 9, /*mean (0.127084), correlation (0.128628)*/
	15, -13, 11, -5, /*mean (0.0197813), correlation (0.166973)*/
	13, -5, 4, 7, /*mean (0.020408), correlation (0.16229)*/
	-1, -12, -2, 12, /*mean (0.0221456), correlation (0.165757)*/
	-9, 7, -14, 12, /*mean (0.0226513), correlation (0.149167)*/
	12, 3, 0, 10, /*mean (0.0229366), correlation (0.164526)*/
	-8, 3, -11, 9, /*mean (0.0255689), correlation (0.168939)*/
	-1, -13, -5, -8, /*mean (0.0271812), correlation (0.167755)*/
	-11, 10, -10, 13, /*mean (0.0295369), correlation (0.156488)*/
	-1, 8, -5, 9, /*mean (0.0315122), correlation (0.168131)*/
	7, -3, 10, -2, /*mean (0.0327181), correlation (0.168398)*/
	-11, -4, -10, -2, /*mean (0.0330985), correlation (0.158571)*/
	4, 2, -2, 3, /*mean (0.0336431), correlation (0.163403)*/
	7, -15, 10, -14, /*mean (0.0341056), correlation (0.165908)*/
	14, -3, 11, -1, /*mean (0.0364397), correlation (0.159289)*/
	10, -15, 7, -13, /*mean (0.0418945), correlation (0.168564)*/
	0, -4, -3, -3, /*mean (0.0433079), correlation (0.156452)*/
	-14, 5, -12, 6, /*mean (0.0465065), correlation (0.166389)*/
	-7, -8, -11, -4, /*mean (0.0471116), correlation (0.161426)*/
	3, -11, 2, -9, /*mean (0.049839), correlation (0.167447)*/
	6, -2, 5, 0, /*mean (0.0509239), correlation (0.15398)*/
	4, -8, 2, -6, /*mean (0.053902), correlation (0.164873)*/
	-14, 9, -13, 10, /*mean (0.0549912), correlation (0.163488)*/
	1, 4, -1, 6, /*mean (0.0588771), correlation (0.160916)*/
	7, 0, 7, 1, /*mean (0.0700893), correlation (0.166851)*/
	3, -9, 3, -8, /*mean (0.0721899), correlation (0.165257)*/
	15, -7, 15, -6, /*mean (0.0724406), correlation (0.158128)*/
	13, 0, 15, 0, /*mean (0.0758682), correlation (0.161292)*/
	-1, 8, 0, 8, /*mean (0.113045), correlation (0.16224)*/
	0, -2, 1, -2, /*mean (0.113477), correlation (0.167667)*/
	14, -7, 15, -7, /*mean (0.118954), correlation (0.147736)*/
	14, -15, 15, -15, /*mean (0.119381), correlation (0.168126)*/
	-10, 15, -9, 15, /*mean (0.125251), correlation (0.162249)*/
	-13, 3, 7, 6, /*mean (0.00479999), correlation (0.219219)*/
	14, -4, 1, -2, /*mean (0.00824922), correlation (0.21777)*/
	-10, -5, 4, 3, /*mean (0.00968426), correlation (0.209761)*/
	15, -15, -13, 8, /*mean (0.010955), correlation (0.219427)*/
	-2, 9, -10, 15, /*mean (0.0132329), correlation (0.209071)*/
	15, -10, 4, 1, /*mean (0.0133064), correlation (0.217042)*/
	-7, -7, -15, 10, /*mean (0.0166562), correlation (0.218543)*/
	3, -9, 10, -3, /*mean (0.0195003), correlation (0.219571)*/
	8, -11, -15, 0, /*mean (0.0202956), correlation (0.212531)*/
	3, 7, 7, 8, /*mean (0.0210477), correlation (0.219609)*/
	15, -2, 0, 15, /*mean (0.0218949), correlation (0.216145)*/
	-9, 14, -15, 15, /*mean (0.0225303), correlation (0.198854)*/
	-1, 4, 4, 9, /*mean (0.022673), correlation (0.216455)*/
	-12, 7, -3, 10, /*mean (0.0232176), correlation (0.219485)*/
	7, -6, 1, -5, /*mean (0.0241555), correlation (0.216268)*/
	-9, -2, -8, 2, /*mean (0.0242852), correlation (0.218637)*/
	15, -11, 10, -9, /*mean (0.0246223), correlation (0.214)*/
	4, -12, -6, -2, /*mean (0.0260055), correlation (0.212964)*/
	-3, 11, 1, 14, /*mean (0.0261957), correlation (0.209467)*/
	1, -6, -9, 5, /*mean (0.0271509), correlation (0.216454)*/
	-1, -7, -1, -3, /*mean (0.0273022), correlation (0.219368)*/
	0, -1, -5, 7, /*mean (0.0289015), correlation (0.218391)*/
	8, -10, 8, -7, /*mean (0.0296319), correlation (0.214665)*/
	7, -7, 6, -2, /*mean (0.0300728), correlation (0.213401)*/
	4, 5, 2, 11, /*mean (0.0308984), correlation (0.208114)*/
	-4, -8, -1, -7, /*mean (0.0317975), correlation (0.213124)*/
	-3, 7, -7, 8, /*mean (0.0326101), correlation (0.202289)*/
	-7, 8, -10, 10, /*mean (0.0341272), correlation (0.211545)*/
	1, -13, -3, -12, /*mean (0.0359383), correlation (0.200318)*/
	-4, -3, -11, 2, /*mean (0.0367855), correlation (0.219147)*/
	-11, 4, -9, 5, /*mean (0.0373517), correlation (0.212568)*/
	7, 1, 8, 3, /*mean (0.0381038), correlation (0.211246)*/
	-4, -2, -3, 0, /*mean (0.0403428), correlation (0.219019)*/
	14, -1, 14, 1, /*mean (0.0415919), correlation (0.217099)*/
	-6, -9, -4, -8, /*mean (0.0417778), correlation (0.205248)*/
	-14, 3, -13, 5, /*mean (0.0421149), correlation (0.214165)*/
	-15, 12, -15, 14, /*mean (0.0423613), correlation (0.21214)*/
	-7, -4, -11, -3, /*mean (0.0431004), correlation (0.214436)*/
	8, -12, 6, -10, /*mean (0.0434246), correlation (0.200937)*/
	-6, 9, -6, 11, /*mean (0.0478334), correlation (0.206306)*/
	14, -4, 15, -3, /*mean (0.053647), correlation (0.188974)*/
	3, 6, 1, 7, /*mean (0.0573556), correlation (0.212569)*/
	9, -14, 9, -13, /*mean (0.0688487), correlation (0.199587)*/
	-14, 7, -14, 8, /*mean (0.0689525), correlation (0.214022)*/
	-6, 15, -3, 15, /*mean (0.0692032), correlation (0.217487)*/
	15, -15, 15, -14, /*mean (0.071628), correlation (0.181987)*/
	11, 4, 10, 5, /*mean (0.0722159), correlation (0.207965)*/
	2, 13, 1, 14, /*mean (0.0743209), correlation (0.216403)*/
	14, 1, 13, 2, /*mean (0.0767025), correlation (0.211112)*/
	8, 5, 10, 5, /*mean (0.0767759), correlation (0.187129)*/
	7, 5, 6, 6, /*mean (0.077005), correlation (0.218884)*/
	4, 3, 3, 4, /*mean (0.0780165), correlation (0.219462)*/
	2, 12, 3, 12, /*mean (0.113101), correlation (0.204923)*/
	10, -2, 11, -2, /*mean (0.116071), correlation (0.206195)*/
	-15, 0, -14, 0, /*mean (0.117125), correlation (0.194557)*/
	-15, 15, -14, 15, /*mean (0.118236), correlation (0.213365)*/
	-1, -7, 0, -7, /*mean (0.118854), correlation (0.203522)*/
	6, -10, 7, -10, /*mean (0.120851), correlation (0.218447)*/
	-5, 12, -4, 12, /*mean (0.121279), correlation (0.20892)*/
	4, -15, -15, 14, /*mean (0.00336927), correlation (0.273755)*/
	6, -14, 8, 5, /*mean (0.00488642), correlation (0.279447)*/
	15, -6, -15, 13, /*mean (0.00646842), correlation (0.255713)*/
	-14, -1, -1, 1, /*mean (0.00707355), correlation (0.283177)*/
	-2, 5, -15, 11, /*mean (0.00939897), correlation (0.275681)*/
	12, -9, -2, -1, /*mean (0.00999546), correlation (0.281584)*/
	14, 1, -15, 9, /*mean (0.0104018), correlation (0.281006)*/
	12, -14, -9, 15, /*mean (0.0105098), correlation (0.278655)*/
	-7, 6, 8, 7, /*mean (0.0107605), correlation (0.280916)*/
	15, -12, 7, 8, /*mean (0.0115775), correlation (0.281633)*/
	-2, -13, 13, -1, /*mean (0.0119449), correlation (0.283968)*/
	-10, -3, -7, 14, /*mean (0.0126407), correlation (0.285558)*/
	14, -14, 3, -4, /*mean (0.0138424), correlation (0.283279)*/
	10, -11, 0, 3, /*mean (0.0141881), correlation (0.283786)*/
	13, -7, -10, -2, /*mean (0.0142357), correlation (0.284256)*/
	3, -13, 6, -6, /*mean (0.0145685), correlation (0.284786)*/
	2, -15, -2, -6, /*mean (0.015031), correlation (0.283388)*/
	-3, -11, 6, -8, /*mean (0.0151909), correlation (0.285178)*/
	12, 0, -9, 14, /*mean (0.0154589), correlation (0.28538)*/
	14, -2, -6, 0, /*mean (0.0154589), correlation (0.284792)*/
	-10, 1, -4, 3, /*mean (0.015688), correlation (0.27994)*/
	-4, 2, -12, 13, /*mean (0.0157788), correlation (0.284405)*/
	-4, -4, 4, -1, /*mean (0.0162499), correlation (0.283384)*/
	1, -10, 2, 2, /*mean (0.0162715), correlation (0.282837)*/
	-9, 2, -2, 15, /*mean (0.0165741), correlation (0.284352)*/
	3, -4, 14, 0, /*mean (0.0168853), correlation (0.285193)*/
	-7, 10, -12, 14, /*mean (0.0170582), correlation (0.280953)*/
	-13, 6, -10, 12, /*mean (0.0173046), correlation (0.282108)*/
	7, -2, -15, 5, /*mean (0.0177498), correlation (0.277614)*/
	1, 0, 7, 4, /*mean (0.0181431), correlation (0.282959)*/
	-1, -14, -15, 6, /*mean (0.0181604), correlation (0.281862)*/
	-5, 11, -11, 12, /*mean (0.0186315), correlation (0.282597)*/
	15, -9, -3, 13, /*mean (0.0188174), correlation (0.285063)*/
	15, -7, 8, -4, /*mean (0.019764), correlation (0.283551)*/
	-14, 2, -4, 8, /*mean (0.0204902), correlation (0.284785)*/
	11, -14, 13, -10, /*mean (0.0206371), correlation (0.26634)*/
	-7, -8, -3, 5, /*mean (0.0209526), correlation (0.279196)*/
	-6, 8, 0, 12, /*mean (0.0210564), correlation (0.279815)*/
	-6, 1, -5, 6, /*mean (0.0210996), correlation (0.274191)*/
	15, -13, 7, -11, /*mean (0.0212898), correlation (0.284267)*/
	-10, 9, -15, 10, /*mean (0.0213935), correlation (0.264998)*/
	12, -7, 9, 0, /*mean (0.0215102), correlation (0.281889)*/
	14, -15, 9, -8, /*mean (0.0222234), correlation (0.285447)*/
	-6, -6, 5, 10, /*mean (0.0222407), correlation (0.282781)*/
	11, -3, -3, 5, /*mean (0.0225779), correlation (0.277698)*/
	6, -15, 3, -11, /*mean (0.0226427), correlation (0.276495)*/
	-4, -1, 0, 9, /*mean (0.0228415), correlation (0.28444)*/
	4, -3, -7, -1, /*mean (0.0232348), correlation (0.283978)*/
	5, -8, 1, 6, /*mean (0.0232392), correlation (0.279056)*/
	11, -2, 10, 4, /*mean (0.0233127), correlation (0.282934)*/
	-9, 0, -15, 8, /*mean (0.0241555), correlation (0.284957)*/
	4, 7, -5, 15, /*mean (0.0241555), correlation (0.268863)*/
	8, -8, -9, 3, /*mean (0.0241771), correlation (0.285399)*/
	2, -6, -10, -4, /*mean (0.0243068), correlation (0.283939)*/
	5, -9, -4, -5, /*mean (0.0244019), correlation (0.2797)*/
	6, 1, -5, 12, /*mean (0.0248644), correlation (0.285081)*/
	7, -3, 0, 8, /*mean (0.0255387), correlation (0.28087)*/
	-4, 4, 0, 5, /*mean (0.0260314), correlation (0.278163)*/
	-7, -6, -5, -2, /*mean (0.0262519), correlation (0.283694)*/
	0, -10, -15, 3, /*mean (0.0263253), correlation (0.278577)*/
	15, -9, 11, -7, /*mean (0.0264377), correlation (0.270823)*/
	10, -10, 6, -8, /*mean (0.026844), correlation (0.273929)*/
	4, -2, 2, 13, /*mean (0.0271077), correlation (0.285601)*/
	1, -6, 6, -5, /*mean (0.0272374), correlation (0.278473)*/
	2, -15, 15, -15, /*mean (0.027756), correlation (0.285406)*/
	6, -5, -2, 2, /*mean (0.0279592), correlation (0.28443)*/
	-4, 1, -11, 7, /*mean (0.0286681), correlation (0.282063)*/
	0, -5, 1, -2, /*mean (0.0287199), correlation (0.274318)*/
	9, 2, 4, 4, /*mean (0.0289058), correlation (0.277918)*/
	9, 0, 13, 1, /*mean (0.0291219), correlation (0.281146)*/
	-6, 11, -7, 15, /*mean (0.0297443), correlation (0.284571)*/
	-1, 0, 2, 1, /*mean (0.0298049), correlation (0.271109)*/
	-9, -1, -15, 2, /*mean (0.0313393), correlation (0.285131)*/
	-12, -2, -10, 0, /*mean (0.0316073), correlation (0.283623)*/
	-1, 6, -1, 10, /*mean (0.0316678), correlation (0.283354)*/
	15, -10, 15, -7, /*mean (0.0320006), correlation (0.273356)*/
	-2, 4, -7, 10, /*mean (0.0322297), correlation (0.270098)*/
	1, 3, -5, 5, /*mean (0.0327268), correlation (0.28388)*/
	9, -8, 11, -6, /*mean (0.0328175), correlation (0.273843)*/
	-7, 6, -6, 9, /*mean (0.0330553), correlation (0.284424)*/
	4, -11, -1, -10, /*mean (0.0330639), correlation (0.276539)*/
	-12, 7, -15, 8, /*mean (0.0338376), correlation (0.282607)*/
	5, 9, 0, 11, /*mean (0.0339197), correlation (0.275682)*/
	-9, -6, -14, 1, /*mean (0.0345421), correlation (0.271055)*/
	-2, 12, -5, 14, /*mean (0.0347194), correlation (0.280668)*/
	-9, 11, -6, 13, /*mean (0.0353677), correlation (0.262672)*/
	10, -5, 12, -3, /*mean (0.035506), correlation (0.281089)*/
	-8, 3, -6, 4, /*mean (0.0360377), correlation (0.284495)*/
	2, 0, -3, 1, /*mean (0.0364915), correlation (0.285368)*/
	-1, 9, 1, 10, /*mean (0.036552), correlation (0.278575)*/
	6, -1, 3, 1, /*mean (0.0367898), correlation (0.285137)*/
	-2, -13, 0, -11, /*mean (0.0371658), correlation (0.279776)*/
	14, -6, 15, -4, /*mean (0.0376197), correlation (0.24281)*/
	8, -4, 5, -1, /*mean (0.0376413), correlation (0.279717)*/
	-7, 6, -10, 8, /*mean (0.0380217), correlation (0.276929)*/
	11, -3, 12, -1, /*mean (0.03867), correlation (0.283306)*/
	4, 3, 5, 5, /*mean (0.0387435), correlation (0.284163)*/
	8, 7, 4, 8, /*mean (0.0387478), correlation (0.283709)*/
	-9, 14, -5, 15, /*mean (0.0388429), correlation (0.279198)*/
	13, 2, 10, 4, /*mean (0.0392492), correlation (0.281706)*/
	-10, 3, -13, 4, /*mean (0.0394307), correlation (0.280171)*/
	-14, 0, -13, 2, /*mean (0.0394567), correlation (0.270968)*/
	-1, -2, -6, 1, /*mean (0.0413758), correlation (0.278449)*/
	1, 10, -1, 13, /*mean (0.0416568), correlation (0.281767)*/
	-11, 5, -10, 7, /*mean (0.0421668), correlation (0.277155)*/
	-1, -4, -3, -1, /*mean (0.0424434), correlation (0.281096)*/
	1, -10, -2, -9, /*mean (0.0427071), correlation (0.273452)*/
	0, 6, -3, 8, /*mean (0.0438655), correlation (0.273137)*/
	-7, 2, -10, 3, /*mean (0.0441292), correlation (0.28543)*/
	-3, -6, -6, -3, /*mean (0.0472931), correlation (0.282788)*/
	15, -12, 13, -10, /*mean (0.0473407), correlation (0.27601)*/
	6, -10, 4, -9, /*mean (0.051395), correlation (0.27438)*/
	1, 0, 0, 2, /*mean (0.052735), correlation (0.276666)*/
	0, -14, -1, -12, /*mean (0.0538113), correlation (0.273652)*/
	4, -12, 2, -11, /*mean (0.053889), correlation (0.28349)*/
	6, 4, 4, 5, /*mean (0.0542867), correlation (0.27425)*/
	-3, -11, -5, -10, /*mean (0.0548529), correlation (0.272518)*/
};


SPHORB::SPHORB(int _nfeatures, int _nlevels, int b): nfeatures(_nfeatures), barrier(b)
{
	initSORB();
	nlevels = min(_nlevels, levels);
}

SPHORB::~SPHORB()
{
	uninitSORB();
}

int SPHORB::descriptorSize() const
{
    return kBytes;
}

int SPHORB::descriptorType() const
{
    return CV_8U;
}

void SPHORB::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors, bool useProvidedKeypoints) const
{
	bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;
	
	_keypoints.clear();
	Mat temp = _image.getMat();
	Mat descriptors;
    if( temp.type() != CV_8UC1 )
        cvtColor(_image, temp, CV_BGR2GRAY);

	// compute how many features should be detected on every scale space level
	vector<int> nfeaturesPerLevel(nlevels);
	float factor = (float)(1.0 / pow(2.0, 1/3.0));
	float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

	int sumFeatures = 0;
	for( int level = 0; level < nlevels-1; level++ )
	{
		nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
		sumFeatures += nfeaturesPerLevel[level];
		ndesiredFeaturesPerScale *= factor;
	}
	nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

	// Gaussian kernel for hexagonal grid
	Mat matKernel(7, 7, CV_64F, kernel);

	// sampling pattern
	const int npoints = 512;
	vector<Point> pattern;
	const Point* pattern0 = (const Point*)bit_pattern;

	std::copy(pattern0, pattern0 + 512, std::back_inserter(pattern));
	

	// detect and describe the features on every level
	for (int l=0;l<nlevels;l++)
	{
		// resize the spherical image
		Size sz(cells[l]*5, cells[l]*5/2);
		Mat image(sz, temp.type());
		resize(temp, image, sz, 0, 0, CV_INTER_AREA);

		// split the spherical image to five parts
		Mat subImg[5];
		for(int i=0;i<5;i++)
		{
			subImg[i].create(Size(2*cells[l]+1, cells[l]+1), image.type());
			splitSphere2(image, subImg[i], i, imgInfos[l][i]);
		}

		// extend each part for boundary pixels 
		extendEdge(subImg[0], subImg[1], subImg[2], subImg[3], subImg[4], SFAST_EDGE + SPHORB_EDGE);

		// the key points on each level
		vector<KeyPoint> levelKeyPoints;

		for (int i=0;i<5;i++)
		{
			xy* corners;
			int* score;
			int cor_num;
			vector<KeyPoint> partKeyPoints;

			// detect the key points and do the non-max suppression
			corners = sfast_corner_detect(&subImg[i].at<uchar>(0,0), &maskes[l].at<uchar>(0,0), 
							maskes[l].cols, (int)maskes[l].step, maskes[l].rows, barrier, &cor_num);
			score = sfastScore(&subImg[i].at<uchar>(0,0), (int)subImg[i].step, corners, cor_num, barrier);
			sfastNonmaxSuppression(corners, score, cor_num, partKeyPoints, i);

			levelKeyPoints.insert(levelKeyPoints.end(), partKeyPoints.begin(), partKeyPoints.end());
			
			delete[] corners;
			delete[] score;
		}

		if (levelKeyPoints.size()>nfeaturesPerLevel[l])
			KeyPointsFilter::retainBest(levelKeyPoints, nfeaturesPerLevel[l]);

		// compute the orientation
		for(size_t i=0;i<levelKeyPoints.size();i++)
			levelKeyPoints[i].angle = IC_Angle(subImg[levelKeyPoints[i].class_id], SPHORB_EDGE, levelKeyPoints[i].pt, geoinfos[l]);

		// filter the image
		for (int i=0;i<5;i++)
			filter2D(subImg[i], subImg[i], -1, matKernel);

		Mat tDesc = Mat::zeros(levelKeyPoints.size(), kBytes, CV_8UC1);

		for(size_t i=0;i<levelKeyPoints.size();i++)
		{
			computeOrbDescriptor(levelKeyPoints[i], subImg[levelKeyPoints[i].class_id], &pattern[0], tDesc.ptr((int)i), kBytes);
		}

		descriptors.push_back(tDesc);

		mappingKeypoint(image, levelKeyPoints, SFAST_EDGE + SPHORB_EDGE, geoinfos[l], l);

		_keypoints.insert(_keypoints.end(), levelKeyPoints.begin(), levelKeyPoints.end());

	}

	descriptors.copyTo(_descriptors);

}

void SPHORB::operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const
{
    (*this)(image, mask, keypoints, noArray(), false);
}

void SPHORB::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
{
    (*this)(image, mask, keypoints, noArray(), false);
}

void SPHORB::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    (*this)(image, Mat(), keypoints, descriptors, true);
}
}

