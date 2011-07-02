/////////////////////////////////////////////////////////////////////
//
//  pwmetrics_cimp.cpp
//
//  The C-mex implementation of some pairwise metrics
//
//  Created by Dahua Lin, on Jul 2, 2007
//  Modified by Dahua Lin, on Jul 30, 2007
//
/////////////////////////////////////////////////////////////////////

#include "mex.h"

#include <math.h>

template<typename Ty>
void cityblock(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2)
{
    const Ty* v2 = X2;
    for(int j = 0; j < n2; ++j)
    {        
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i)
        {
            Ty s = fabs(*v1 - *v2);
            for (int k = 1; k < d; ++k) 
            {
                s += fabs(v1[k] - v2[k]);
            }
            *M++ = s;           
            
            v1 += d;
        }        
        v2 += d;
    }
}

template<typename Ty>
void mindiff(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2)
{
    const Ty* v2 = X2;
    for(int j = 0; j < n2; ++j)
    {        
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i)
        {
            Ty s = fabs(*v1 - *v2);
            for (int k = 1; k < d; ++k) 
            {
                Ty cv = fabs(v1[k] - v2[k]);
                if (cv < s) s = cv;
            }
            *M++ = s;           
            
            v1 += d;
        }        
        v2 += d;
    }
}

template<typename Ty>
void maxdiff(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2)
{
    const Ty* v2 = X2;
    for(int j = 0; j < n2; ++j)
    {        
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i)
        {
            Ty s = fabs(*v1 - *v2);
            for (int k = 1; k < d; ++k) 
            {
                Ty cv = fabs(v1[k] - v2[k]);
                if (cv > s) s = cv;
            }
            *M++ = s;           
            
            v1 += d;
        }        
        v2 += d;
    }
}

template<typename Ty>
void minkowski(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2, Ty p)
{
    const Ty* v2 = X2;
    for(int j = 0; j < n2; ++j)
    {        
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i)
        {
            Ty s = pow(fabs(*v1 - *v2), p);
            for (int k = 1; k < d; ++k) 
            {
                s += pow(fabs(v1[k] - v2[k]), p);
            }
            *M++ = pow(s, 1/p);           
            
            v1 += d;
        }        
        v2 += d;
    }
}

template<typename Ty>
void column_sum(const Ty* X, int d, int n, Ty* s)
{
    for (int i = 0; i < n; ++i)
    {
        Ty cs = 0;
        for (int k = 0; k < d; ++k)
        {
            cs += *X++;
        }
        s[i] = cs;
    }
}



template<typename Ty>
void intersect(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2)
{
    // compute the respective sums
    Ty* hs1 = new Ty[n1];
    Ty* hs2 = new Ty[n2];    
    column_sum(X1, d, n1, hs1);
    column_sum(X2, d, n2, hs2);
    
    // compute metrics
    const Ty* v2 = X2;
    for (int j = 0; j < n2; ++j, v2 += d)
    {
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i, v1 += d)
        {
            Ty s = 0;
            
            for (int k = 0; k < d; ++k)
            {
                Ty u1 = v1[k];
                Ty u2 = v2[k];
                s += ((u1 < u2) ? u1 : u2);
            }
            
            Ty smax = ((hs1[i] > hs2[j]) ? hs1[i] : hs2[j]);
            
            *M++ = ((smax > 0) ? s / smax : 1);
        }
    }
    
    // release resources
    delete[] hs1;
    delete[] hs2;
}


template<typename Ty>
void chisq(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2)
{
    // compute metrics
    const Ty* v2 = X2;
    for (int j = 0; j < n2; ++j, v2 += d)
    {
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i, v1 += d)
        {
            Ty s = 0;
            
            for (int k = 0; k < d; ++k)
            {
                Ty u1 = v1[k];
                Ty u2 = v2[k];                
                Ty sr = u1 + u2;
                Ty dr = u1 - u2;
                              
                if (dr != 0 && sr > 0)
                {
                    s += dr * dr / (2 * sr);
                }
            }
            
            *M++ = s;
        }
    }
}


template<typename Ty>
void sum_xlogx(const Ty* X, int d, int n, Ty* s)
{
    for (int i = 0; i < n; ++i)
    {
        Ty cs = 0;
        for (int k = 0; k < d; ++k)
        {
            Ty u = *X++;
            
            cs += (u > 0 ? u * log(u) : 0);
        }
        *s++ = cs;
    }    
}

template<typename Ty>
void kldiv(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2)
{
    // compute xlogx
    Ty* xs = new Ty[n1];
    sum_xlogx(X1, d, n1, xs);
    
    // compute metrics
    const Ty* v2 = X2;
    for (int j = 0; j < n2; ++j, v2 += d)
    {
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i, v1 += d)
        {
            Ty sa = 0;
            
            for (int k = 0; k < d; ++k)
            {
                Ty u1 = v1[k];
                Ty u2 = v2[k];
                sa += (u1 > 0 ? u1 * log(u2) : 0);
            }
            
            *M++ = xs[i] - sa;
        }
    }
    
    // release memory
    delete[] xs;
    
}


template<typename Ty>
void jeffrey(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2)
{
    // compute xlogx and ylogy
    Ty* xs1 = new Ty[n1];
    Ty* xs2 = new Ty[n2];
    
    sum_xlogx(X1, d, n1, xs1);
    sum_xlogx(X2, d, n2, xs2);    
    
    // compute metrics
    const Ty* v2 = X2;
    for (int j = 0; j < n2; ++j, v2 += d)
    {
        const Ty* v1 = X1;
        for (int i = 0; i < n1; ++i, v1 += d)
        {
            Ty sa = 0;
            
            for (int k = 0; k < d; ++k)
            {             
                Ty ua = (v1[k] + v2[k]) * 0.5;                                
                sa += (ua > 0 ? ua * log(ua) : 0);
            }
            
            *M++ = xs1[i] + xs2[j] - 2 * sa;
        }
    }
    
    // release memory
    delete[] xs1;
    delete[] xs2;
}



template<typename Ty>
void compute_metrics(const Ty* X1, const Ty* X2, Ty* M, int d, int n1, int n2, int opcode, const mxArray *prhs[])
{
    switch (opcode)
    {
        case 1:
            cityblock(X1, X2, M, d, n1, n2);
            break;
        case 2:
            mindiff(X1, X2, M, d, n1, n2);
            break;
        case 3:
            maxdiff(X1, X2, M, d, n1, n2);
            break;
        case 4:
            minkowski(X1, X2, M, d, n1, n2, *((const Ty*)mxGetData(prhs[3])));
            break;
        case 5:
            intersect(X1, X2, M, d, n1, n2);
            break;
        case 6:
            chisq(X1, X2, M, d, n1, n2);
            break;
        case 7:
            kldiv(X1, X2, M, d, n1, n2);
            break;
        case 8:
            jeffrey(X1, X2, M, d, n1, n2);
            break;
    }
}


/**
 * main entry
 * Input:   X1, X2, opcode, extra parameters
 * Output:  D
 *
 * opcodes:
 *      1 - cityblock
 *      2 - min diff
 *      3 - max diff
 *      4 - minkowski
 *      5 - intersect
 *      7 - jeffrey
 *
 * No input checking. (All necessary checking should be performed in slmetric_pw.m)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray* mxX1 = prhs[0];
    const mxArray* mxX2 = prhs[1];
    const mxArray* mxOpcode = prhs[2];
    
    mwSize d = mxGetM(mxX1);
    mwSize n1 = mxGetN(mxX1);
    mwSize n2 = mxGetN(mxX2);
    
    int opcode = *((const int*)mxGetData(mxOpcode));
            
    if (mxGetClassID(mxX1) == mxDOUBLE_CLASS)
    {
        mxArray* mxD = mxCreateNumericMatrix(n1, n2, mxDOUBLE_CLASS, mxREAL);
        compute_metrics((const double*)mxGetData(mxX1), (const double*)mxGetData(mxX2), 
                        (double *)mxGetData(mxD), (int)d, (int)n1, (int)n2, 
                        opcode, prhs);
        plhs[0] = mxD;
    }
    else // SINGLE
    {
        mxArray* mxD = mxCreateNumericMatrix(n1, n2, mxSINGLE_CLASS, mxREAL);
        compute_metrics((const float*)mxGetData(mxX1), (const float*)mxGetData(mxX2), 
                        (float*)mxGetData(mxD), (int)d, (int)n1, (int)n2, 
                        opcode, prhs);
        plhs[0] = mxD;
    }        
}




