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

#include "detector.h"

#define Compare(X, Y) ((X)>=(Y))

static void makeOffsets(int pixel[], int xstride)
{
	pixel[0] = 0 + 3 * xstride;		
	pixel[1] = 1 + 2 * xstride;		
	pixel[2] = 2 + 1 * xstride;		
	pixel[3] = 3 + 0 * xstride;		
	pixel[4] = 3 + -1 * xstride;		
	pixel[5] = 3 + -2 * xstride;		
	pixel[6] = 3 + -3 * xstride;		
	pixel[7] = 2 + -3 * xstride;		
	pixel[8] = 1 + -3 * xstride;		
	pixel[9] = 0 + -3 * xstride;		
	pixel[10] = -1 + -2 * xstride;		
	pixel[11] = -2 + -1 * xstride;		
	pixel[12] = -3 + 0 * xstride;		
	pixel[13] = -3 + 1 * xstride;		
	pixel[14] = -3 + 2 * xstride;		
	pixel[15] = -3 + 3 * xstride;		
	pixel[16] = -2 + 3 * xstride;		
	pixel[17] = -1 + 3 * xstride;
}

int* sfastScore(const unsigned char* i, int stride, xy* corners, int num_corners, int b)
{	
	int* scores = (int*)malloc(sizeof(int)* num_corners);

	int pixel[18];
	makeOffsets(pixel, stride);

	for(int n=0; n < num_corners; n++)
		scores[n] = sfast_corner_score(i + corners[n].y*stride + corners[n].x, pixel, b);

	return scores;
}

void sfastNonmaxSuppression(const xy* corners, const int* scores, int num_corners, vector<KeyPoint>& kps, int partIndex)
{
	bool goto_enabled = false;
	int num_nonmax=0;
	int last_row;
	int* row_start;
	int i, j;
	const int sz = (int)num_corners; 
	kps.clear();

	/*Point above points (roughly) to the pixel above the one of interest, if there
    is a feature there.*/
	int point_above = 0;
	int point_below = 0;

	
	if(num_corners < 1)
		return;

	/* Find where each row begins
	   (the corners are output in raster scan order). A beginning of -1 signifies
	   that there are no corners on that row. */
	last_row = corners[num_corners-1].y;
	row_start = (int*)malloc((last_row+1)*sizeof(int));

	for(i=0; i < last_row+1; i++)
		row_start[i] = -1;
	
	int prev_row = -1;
	for(i=0; i< num_corners; i++)
	{
		if(corners[i].y != prev_row)
		{
			row_start[corners[i].y] = i;
			prev_row = corners[i].y;
		}
	}
	
	
	
	for(i=0; i < sz; i++)
	{
		int score = scores[i];
		xy pos = corners[i];
			
		/*Check left */
		if(i > 0)
			if(corners[i-1].x == pos.x-1 && corners[i-1].y == pos.y && Compare(scores[i-1], score))
				continue;
			
		/*Check right*/
		if(i < (sz - 1))
			if(corners[i+1].x == pos.x+1 && corners[i+1].y == pos.y && Compare(scores[i+1], score))
				continue;
			
		/*Check above (if there is a valid row above)*/
		if(pos.y != 0 && row_start[pos.y - 1] != -1) 
		{
			/*Make sure that current point_above is one
			  row above.*/
			if(corners[point_above].y < pos.y - 1)
				point_above = row_start[pos.y-1];
			
			/*Make point_above point to the first of the pixels above the current point,
			  if it exists.*/
			for(; corners[point_above].y < pos.y && corners[point_above].x < pos.x; point_above++)
			{}
			
			
			for(j=point_above; corners[j].y < pos.y && corners[j].x <= pos.x + 1; j++)
			{
				int x = corners[j].x;
				if( (x ==pos.x || x == pos.x+1) && Compare(scores[j], score)){
                    goto_enabled = true;
					goto cont;
				}
			}
			
		}
			
		/*Check below (if there is anything below)*/
		if(pos.y != last_row && row_start[pos.y + 1] != -1 && point_below < sz) /*Nothing below*/
		{
			if(corners[point_below].y < pos.y + 1)
				point_below = row_start[pos.y+1];
			
			/* Make point below point to one of the pixels belowthe current point, if it
			   exists.*/
			for(; point_below < sz && corners[point_below].y == pos.y+1 && corners[point_below].x < pos.x - 1; point_below++)
			{}

			for(j=point_below; j < sz && corners[j].y == pos.y+1 && corners[j].x <= pos.x; j++)
			{
				int x = corners[j].x;
				if( (x == pos.x - 1 || x ==pos.x) && Compare(scores[j],score)){
					goto_enabled = true;
					goto cont;
				}
			}
		}
		
		cont:

		if (!goto_enabled){ // If this part reached by goto
			KeyPoint kp;
			kp.pt.x = corners[i].x;
			kp.pt.y = corners[i].y;
			kp.response = score;
			kp.class_id = partIndex;
			kps.push_back(kp);
		}
		goto_enabled = false;

	}

	free(row_start);
}