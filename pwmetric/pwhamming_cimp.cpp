/////////////////////////////////////////////////////////////////////
//
//  pwhamming_cimp.cpp
//
//  The implementation of pairwise hamming metric
//
//  Created by Dahua Lin, on Jul 30, 2007
//
/////////////////////////////////////////////////////////////////////

#include "mex.h"


// The core function to compute hamming distances
//  X1: d x n1 (logical)
//  X2: d x n2 (logical)
//  dists: n1 x n2 (double)
void compute_hamming(const mxLogical* X1, const mxLogical* X2, double* dists, int d, int n1, int n2)
{
    const mxLogical* xc2 = X2;
    for (int j = 0; j < n2; ++j, xc2 += d)
    {        
        const mxLogical* xc1 = X1;
        for (int i = 0; i < n1; ++i, xc1 += d)
        {
            int s = 0;
            for (int k = 0; k < d; ++k)
            {
                if (xc1[k] != xc2[k]) ++s;
            }
            *dists ++ = (double)s;
        }
    }
}


// The main entry
// Input
//   - X1:  d x n1 (logical)
//   - X2:  d x n2 (logical)
// Output
//   - D:   n1 x n2 (double)
//
// No argument checking in the function
//
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray* mxX1 = prhs[0];
    const mxArray* mxX2 = prhs[1];
    
    int d = (int)mxGetM(mxX1);
    int n1 = (int)mxGetN(mxX1);
    int n2 = (int)mxGetN(mxX2);
    
    const mxLogical* X1 = (const mxLogical*)mxGetData(mxX1);
    const mxLogical* X2 = (const mxLogical*)mxGetData(mxX2);
    
    mxArray* mxD = mxCreateDoubleMatrix((mwSize)n1, (mwSize)n2, mxREAL);
    double* dists = mxGetPr(mxD);
    
    compute_hamming(X1, X2, dists, d, n1, n2);
    
    plhs[0] = mxD;
}


