// Tomasz Malisiewicz
// tomasz@cmu.edu
// Fast way to find K smallest elements in a vector
#include "mex.h"

#include <queue>
#include <utility>
#include <iostream>

using std::priority_queue;
using std::pair;

// Type templated function to find and sort the K smallesest
// elements
template<class T>
void psort(T* in, int m, int K, T* out0, int* out1)
{
  priority_queue<pair<T,int> > q;

  // insert first K elements into priority queue
  for (int i = 0; i < K; ++i)
  {
    q.push(pair<T,int>(in[i],i));
  }
  T curtop = q.top().first;
  //std::cout<<"top is now " << curtop << std::endl;
  //std::cout<<"k/m are " << K << " " << m << std::endl;

  for (int i = K; i < m; ++i)
  {
    T& curval = in[i];
    //std::cout<<"curval is now " << curval << std::endl;
    if (curval < curtop)
    {
      //std::cout<<"  -popping " << q.top().first << std::endl;
      q.pop();
      //std::cout<<"  -adding " << curval << std::endl;
      q.push(pair<T,int>(curval,i));
      curtop = q.top().first;
      //std::cout<<"top is now " << curtop << std::endl;
    }
  }

  for (int i = 0; i < K; ++i)
  {
    out0[(K-1)-i] = q.top().first;
    // matlab counts at 1, c++ at 0
    out1[(K-1)-i] = q.top().second+1;;
    q.pop();
  }

  if (q.size() != 0)
    mexErrMsgTxt("queue not empty, bug");
  
}



void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

  // make sure number of args is correct
  if (nrhs != 2 || nlhs != 2) 
    mexErrMsgTxt("Usage [Ksmallest,Ksmallest_inds] = psort(single_vector,K);");

  // make sure we are dealing with second argument a scalar
  if (mxGetN(prhs[1])*mxGetM(prhs[1]) != 1) {
    mexErrMsgTxt("psort.cpp: Second input argument must be a scalar");
  }
      
  
  int K = (int)mxGetScalar(prhs[1]);

  int M = mxGetM(prhs[0]);
  int N = mxGetN(prhs[0]);

  // make sure we have been passed a vector
  if ( (N != 1) )
      mexErrMsgTxt("psort.cpp: First argument should be a column vector");

  // make sure K isn't negative, 0 or just simply too big
  if ((K > M) || (K < 1))
    mexErrMsgTxt("psort.cpp: K > M or K<1");

  // create the inds
  plhs[1] = mxCreateNumericMatrix(K, 1, mxINT32_CLASS, mxREAL);
  int *out1 = (int*)mxGetPr(plhs[1]);

  // read class information so that we can work with singles and doubles
  mxClassID category = mxGetClassID(prhs[0]);

  
  switch (category)  
  { 
  case mxSINGLE_CLASS: 
  { 
    //std::cout<<"About to process single with K=" << K << " N=" << N << std::endl;
    plhs[0] = mxCreateNumericMatrix(K, 1, mxSINGLE_CLASS, mxREAL);
    typedef float T;
    T *out0 = (T*)mxGetPr(plhs[0]);
    T *in = (T*)mxGetPr(prhs[0]);
    psort<T>(in, M, K, out0, out1);
    break;
  }
  case mxDOUBLE_CLASS: 
  {
    plhs[0] = mxCreateNumericMatrix(K, 1, mxDOUBLE_CLASS, mxREAL);
    typedef double T; 
    T *out0 = (T*)mxGetPr(plhs[0]);
    T *in = (T*)mxGetPr(prhs[0]);
    psort<T>(in, M, K, out0, out1);
    break;
  }
  default:
  {
    mexErrMsgTxt("psort.cpp: invalid data type: double or single only!");
  }
  }
}
