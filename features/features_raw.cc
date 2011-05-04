#include <math.h>
#include <iostream>
#include "mex.h"

// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
double uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
double vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a double color image and a bin size 
// returns HOG features
mxArray *process(const mxArray *mximage, const mxArray *mxsbin) {
  double *im = (double *)mxGetPr(mximage);
  const int *dims = mxGetDimensions(mximage);
  if (mxGetNumberOfDimensions(mximage) != 3 ||
      dims[2] != 3 ||
      mxGetClassID(mximage) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Invalid input");

  int sbin = (int)mxGetScalar(mxsbin);

  // memory for caching orientation histograms & their norms
  int blocks[2];
  blocks[0] = (int)round((double)dims[0]/(double)sbin);
  blocks[1] = (int)round((double)dims[1]/(double)sbin);
  double *hist = (double *)mxCalloc(blocks[0]*blocks[1]*18, sizeof(double));
  double *norm = (double *)mxCalloc(blocks[0]*blocks[1], sizeof(double));

  // memory for HOG features
  int out[3];
  out[0] = max(blocks[0]-2, 0);
  out[1] = max(blocks[1]-2, 0);
  out[2] = 18;
  mxArray *mxfeat = mxCreateNumericArray(3, out, mxDOUBLE_CLASS, mxREAL);
  double *feat = (double *)mxGetPr(mxfeat);
  
  int visible[2];
  visible[0] = blocks[0]*sbin;
  visible[1] = blocks[1]*sbin;
  
  for (int x = 1; x < visible[1]-1; x++) {
    for (int y = 1; y < visible[0]-1; y++) {
      // first color channel
      double *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
      double dy = *(s+1) - *(s-1);
      double dx = *(s+dims[0]) - *(s-dims[0]);
      double v = dx*dx + dy*dy;

      // second color channel
      s += dims[0]*dims[1];
      double dy2 = *(s+1) - *(s-1);
      double dx2 = *(s+dims[0]) - *(s-dims[0]);
      double v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += dims[0]*dims[1];
      double dy3 = *(s+1) - *(s-1);
      double dx3 = *(s+dims[0]) - *(s-dims[0]);
      double v3 = dx3*dx3 + dy3*dy3;

      // pick channel with strongest gradient
      if (v2 > v) {
	v = v2;
	dx = dx2;
	dy = dy2;
      } 
      if (v3 > v) {
	v = v3;
	dx = dx3;
	dy = dy3;
      }

      // snap to one of 18 orientations
      double best_dot = 0;
      int best_o = 0;
      for (int o = 0; o < 9; o++) {
	double dot = uu[o]*dx + vv[o]*dy;
	if (dot > best_dot) {
	  best_dot = dot;
	  best_o = o;
	} else if (-dot > best_dot) {
	  best_dot = -dot;
	  best_o = o+9;
	}
      }
      
      // add to 4 histograms around pixel using linear interpolation
      double xp = ((double)x+0.5)/(double)sbin - 0.5;
      double yp = ((double)y+0.5)/(double)sbin - 0.5;
      int ixp = (int)floor(xp);
      int iyp = (int)floor(yp);
      double vx0 = xp-ixp;
      double vy0 = yp-iyp;
      double vx1 = 1.0-vx0;
      double vy1 = 1.0-vy0;
      v = sqrt(v);

      // if (ixp >= 0 || iyp < 0) {
      //   std::cout<<"ixp/iyp are " << ixp << "," << iyp << std::endl;
      // }
      // this throws bug sometimes
      if (ixp >= 0 && iyp >= 0) {
        *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += v;
      }

      // if (ixp >= 0 && iyp >= 0) {
      //   *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
      //     vx1*vy1*v;
      // }

      // if (ixp+1 < blocks[1] && iyp >= 0) {
      //   *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
      //     vx0*vy1*v;
      // }

      // if (ixp >= 0 && iyp+1 < blocks[0]) {
      //   *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
      //     vx1*vy0*v;
      // }

      // if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
      //   *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
      //     vx0*vy0*v;
      // }


    }
  }

  // compute features
  for (int x = 0; x < out[1]; x++) {
    for (int y = 0; y < out[0]; y++) {
      double *dst = feat + x*out[0] + y;      
      double *src, *p, n1, n2, n3, n4;

      // contrast-sensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (int o = 0; o < 18; o++) {
        *dst = *src;
	dst += out[0]*out[1];
	src += blocks[0]*blocks[1];
      }

    }
  }

  mxFree(hist);
  mxFree(norm);
  return mxfeat;
}

// matlab entry point
// F = features(image, bin)
// image should be color with double values
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 2)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 1)
    mexErrMsgTxt("Wrong number of outputs");
  plhs[0] = process(prhs[0], prhs[1]);
}



