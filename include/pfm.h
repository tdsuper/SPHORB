/*
 * File name: pfm.h
 *
 * PFM -- portable float maps, one High Dynamic Resolution (HDR) image format
 *    PF
 *    wid hei
 *    1 (to max)
 *    data: 12 bytes for one pixel, 4 bytes one channel, i.e. float32
 *
 * Author: 
 *  Liang Wan
 * Date: 28 June 2004
 * Last update date: 16 Dec 2004
 */

#ifndef _PFM_H
#define _PFM_H

bool get_pfm_size(const char *filename, int& w, int& h);

bool read_pfm(const char *filename, float *pimg);
bool read_pfm2(const char *file, float *pimg, int option);

bool write_pfm(const char *filename, float*img, int h, int w, int option); 
// write in normal pfm format, i.e. bottom-up
bool write_pfm2(const char*filename, float*pimg, int h, int w, int option);

#endif //_PFM_H