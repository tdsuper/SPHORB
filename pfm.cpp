/*
 * File name: pfm.cpp
 *
 */

#include <stdio.h>
#include <string.h>
#include "pfm.h"

bool get_pfm_size(const char *filename, int& w, int& h)
{
  FILE *fp = fopen(filename,"rb");
  if ( fp == NULL ) 
  {
    printf("Error reading file %s\n",filename);
    return false;
  }
  char buff[128];

  fscanf(fp, "%s\r", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  if(strncmp(buff, "PF", 2))
    { fclose(fp); return false; }

  fscanf(fp, "%s", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  sscanf(buff, "%d", &w);    

  fscanf(fp, "%s\r", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  sscanf(buff, "%d", &h);    
        
  fclose(fp);
  return true;
}

// pimg should be allocated outside
bool read_pfm(const char *file, float * pimg)
{
	FILE *fp = fopen(file,"rb");
	if ( fp == NULL ) return false;
	char buff[128];

	fscanf(fp, "%s\r", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  if(strncmp(buff, "PF", 2))
  { fclose(fp); return false; }
  
  int w, h;
  fscanf(fp, "%s", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  sscanf(buff, "%d", &w);    
  
  fscanf(fp, "%s\r", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  sscanf(buff, "%d", &h);    

  float tomin;
	fscanf(fp, "%f", &tomin);
  buff[0] = fgetc(fp);
  
  if ( tomin < 0 )
  {
    int index = w*(h-1)*3;
    for (int i=0; i<h; i++)
    {
      for (int j=0; j<w*3; j+=3)
      {
        fread(pimg+index+j, sizeof(float), 1, fp);
        fread(pimg+index+j+1, sizeof(float), 1, fp);
        fread(pimg+index+j+2, sizeof(float), 1, fp);
      }
      index -= w*3;
    }
  }
  else
	  fread(pimg, sizeof(float), w * h * 3, fp);

  fclose(fp);
  return true;
}

// pimg should be allocated outside
// Note: only if w = h, can we rotate them by time of 90 degree.
bool read_pfm2(const char *file, float* pimg, int option)
{
	FILE *fp = fopen(file,"rb");
  if ( fp == NULL )
  {
      printf("Cubic files %s don't exist!\n",file);
      return false;
  }
	char buff[128];

  fscanf(fp, "%s\r", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  if(strncmp(buff, "PF", 2))
  { fclose(fp); return false; }

  int w, h;
  fscanf(fp, "%s", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  sscanf(buff, "%d", &w);    
  
  fscanf(fp, "%s\r", buff);
  while(buff[0] == '#') fgets(buff, 128, fp);
  sscanf(buff, "%d", &h);    

  float tomin;
  fscanf(fp, "%f", &tomin);
  buff[0] = fgetc(fp);


  float *img = new float [w*h*3];
  int index = w*(h-1)*3;
  if ( tomin < 0 )
  {
    for (int i=0; i<h; i++)
    {
      for (int j=0; j<w*3; j+=3)
      {
        fread(img+index+j, sizeof(float), 1, fp);
        fread(img+index+j+1, sizeof(float), 1, fp);
        fread(img+index+j+2, sizeof(float), 1, fp);
      }
      index -= w*3;
    }
  }
  else
    fread(img, sizeof(float), w * h * 3, fp);
  fclose(fp);


	int i,j;
	switch (option)
	{
	case 0:
    memcpy(pimg, img, w*h*3*sizeof(float));
    break;
	case 90: // rotate 90 degree in counter-clockwise direction (to be check)
    index = 0;
    for ( i=0; i<h; i++)
    {
			for ( j=w-1; j>=0; j--)
      {
			  memcpy(pimg+(j*h+i)*3, img+index, 3*sizeof(float));
        index += 3;
      }
    }
		break;
	case 180:
    index = 0;
    for ( i=h-1; i>=0; i--)
    {
      for ( j=w-1; j>=0; j--)
      {
        memcpy(pimg+(i*w+j)*3, img+index, 3*sizeof(float));
        index += 3;
      }
    }
		break;
	case 270:
    index = 0;
    for ( i=h-1; i>=0; i--)
    {
		  for ( j=0; j<w; j++)
      {
			  memcpy(pimg+(j*h+i)*3, img+index, 3*sizeof(float));
        index += 3;
      }
    }
		break;
	case -180: // upside down
    index = 0;
    for ( i=h-1; i>=0; i-- )
    {
		  for ( j=0; j<w; j++ )
      {
        memcpy(pimg+(i*w+j)*3, img+index, 3*sizeof(float));
        index += 3;
      }
    }
    break;
	case -90: // leftside right
    index = 0;
    for ( i=0; i<h; i++ )
    {
      for ( j=w-1; j>=0; j-- )
      {       
        memcpy(pimg+(i*w+j)*3, img+index, 3*sizeof(float));
        index += 3;
      }
    }
	}

  delete [] img;
  return true;
}

// save pfm in top-down order
// Note w=h for rotation by a 90 time degree
bool write_pfm(const char*filename, float*img, int h, int w, int option)
{
	FILE *fp = fopen(filename,"wb");
  
  if ( fp == NULL )
  {
     fclose(fp);
     return false;
  }

	fprintf(fp, "PF\r");
	fprintf(fp, "%d %d\r", w, h);
	fprintf(fp, "%f\r", 1.0);	
	
	int i,j;
	switch (option)
	{
	case 0:
		fwrite(img, sizeof(float), h*w*3, fp);
		break;
	case 90:
		for ( i=0; i<h; i++)
			for ( j=w-1; j>=0; j--)
				fwrite(img+(j*h+i)*3, sizeof(float), 3, fp);
		break;
	case 180:
		for ( i=h-1; i>=0; i--)
			for ( j=w-1; j>=0; j--)
				fwrite(img+(i*w+j)*3, sizeof(float), 3, fp);
		break;
	case 270:
		for ( i=h-1; i>=0; i--)
			for ( j=0; j<w; j++)
				fwrite(img+(j*h+i)*3, sizeof(float), 3, fp);
		break;
	case -180: // upside down
		for ( i=h-1; i>=0; i-- )
			for ( j=0; j<w; j++ )
				fwrite(img+(i*w+j)*3, sizeof(float), 3, fp);
		break;
	}
	fclose(fp);

  return true;
}

// write in normal pfm format, i.e. bottom-up
bool write_pfm2(const char*filename, float*pimg, int h, int w, int option)
{

  int i,j, index;
  float *img;
  if (option == 0)
    img = pimg;
  else
  {
    img = new float [w*h*3];
    switch (option)
    {
    case 0:
      break;
    case 90: // rotate 90 degree in counter-clockwise direction
      index = 0;
      for ( i=0; i<h; i++)
      {
			  for ( j=w-1; j>=0; j--)
        {
			    memcpy(img+(j*h+i)*3, pimg+index, 3*sizeof(float));
          index += 3;
        }
      }
      j=h; h=w; w=j;
		  break;
	  case 180:
      index = 0;
      for ( i=h-1; i>=0; i--)
      {
        for ( j=w-1; j>=0; j--)
        {
          memcpy(img+(i*w+j)*3, pimg+index, 3*sizeof(float));
          index += 3;
        }
      }
		  break;
	  case 270:
      index = 0;
      for ( i=h-1; i>=0; i--)
      {
		    for ( j=0; j<w; j++)
        {
			    memcpy(img+(j*h+i)*3, pimg+index, 3*sizeof(float));
          index += 3;
        }
      }
      j=h; h=w; w=j;
		  break;
	  case -180: // upside down
      index = 0;
      for ( i=h-1; i>=0; i-- )
      {
		    for ( j=0; j<w; j++ )
        {
          memcpy(img+(i*w+j)*3, pimg+index, 3*sizeof(float));
          index += 3;
        }
      }
      break;
	  case -90: // leftside right
      index = 0;
      for ( i=0; i<h; i++ )
      {
        for ( j=w-1; j>=0; j-- )
        {        
          memcpy(img+(i*w+j)*3, pimg+index, 3*sizeof(float));
          index += 3;
        }
      }
      break;
    }
  }
  
  FILE *fp = fopen(filename,"wb");
  if ( fp == NULL )
  {
     fclose(fp);
     return false;
  }


	fprintf(fp, "PF\r");
	fprintf(fp, "%d %d\r", w, h);
	fprintf(fp, "%f\r", -1.0);

  index = w*(h-1)*3;
  for (i=0; i<h; i++)
  {
    for (j=0; j<w*3; j+=3)
    {
      fwrite(img+index+j, sizeof(float), 1, fp);
      fwrite(img+index+j+1, sizeof(float), 1, fp);
      fwrite(img+index+j+2, sizeof(float), 1, fp);
    }
    index -= w*3;
  }
  fclose(fp);

  if (option != 0)
  {
    delete [] img;
  }

  return true;
}


