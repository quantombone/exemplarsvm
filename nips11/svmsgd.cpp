// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA



// $Id: svmsgd.cpp,v 1.13 2007/10/02 20:40:06 cvs Exp $


#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>

using namespace std;


typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;


// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

// Select loss
#define LOSS HINGELOSS

// Zero when no bias
// One when bias term
#define BIAS 1


inline 
double loss(double z)
{
#if LOSS == LOGLOSS
  if (z > 18)
    return exp(-z);
  if (z < -18)
    return -z;
  return log(1+exp(-z));
#elif LOSS == LOGLOSSMARGIN
  if (z > 18)
    return exp(1-z);
  if (z < -18)
    return 1-z;
  return log(1+exp(1-z));
#elif LOSS == SMOOTHHINGELOSS
  if (z < 0)
    return 0.5 - z;
  if (z < 1)
    return 0.5 * (1-z) * (1-z);
  return 0;
#elif LOSS == SQUAREDHINGELOSS
  if (z < 1)
    return 0.5 * (1 - z) * (1 - z);
  return 0;
#elif LOSS == HINGELOSS
  if (z < 1)
    return 1 - z;
  return 0;
#else
# error "Undefined loss"
#endif
}

inline 
double dloss(double z)
{
#if LOSS == LOGLOSS
  if (z > 18)
    return exp(-z);
  if (z < -18)
    return 1;
  return 1 / (exp(z) + 1);
#elif LOSS == LOGLOSSMARGIN
  if (z > 18)
    return exp(1-z);
  if (z < -18)
    return 1;
  return 1 / (exp(z-1) + 1);
#elif LOSS == SMOOTHHINGELOSS
  if (z < 0)
    return 1;
  if (z < 1)
    return 1-z;
  return 0;
#elif LOSS == SQUAREDHINGELOSS
  if (z < 1)
    return (1 - z);
  return 0;
#else
  if (z < 1)
    return 1;
  return 0;
#endif
}


// -- stochastic gradient

class SvmSgd
{
public:
  SvmSgd(int dim, double lambda);
  void train(int imin, int imax, const xvec_t &x, const yvec_t &y,
             const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, 
            const char *prefix = "");
private:
  double  t;
  double  lambda;
  FVector w;
  double  wscale;
  double  bias;
};



SvmSgd::SvmSgd(int dim, double l)
  : lambda(l), w(dim), wscale(1), bias(0)
{
  // Shift t in order to have a 
  // reasonable initial learning rate.
  // This assumes |x| \approx 1.
  double maxw = 1.0 / sqrt(lambda);
  double typw = sqrt(maxw);
  double eta0 = typw / max(1.0,dloss(-typw));
  t = 1 / (eta0 * lambda);
}

void 
SvmSgd::train(int imin, int imax, 
              const xvec_t &xp, const yvec_t &yp,
              const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  for (int i=imin; i<=imax; i++)
    {
      double eta = 1.0 / (lambda * t);
      double s = 1 - eta * lambda;
      wscale *= s;
      if (wscale < 1e-9)
        {
          w.scale(wscale);
          wscale = 1;
        }
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double wx = dot(w,x) * wscale;
      double z = y * (wx + bias);
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        {
          double etd = eta * dloss(z);
          w.add(x, etd * y / wscale);
#if BIAS
          // Slower rate on the bias because
          // it learns at each iteration.
          bias += etd * y * 0.01;
#endif
        }
      t += 1;
    }
  double wnorm =  dot(w,w) * wscale * wscale;
  cout << prefix << setprecision(6) 
       << "Norm: " << wnorm << ", Bias: " << bias << endl;
}


void 
SvmSgd::test(int imin, int imax, 
             const xvec_t &xp, const yvec_t &yp, 
             const char *prefix)

{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  int nerr = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double wx = dot(w,x) * wscale;
      double z = y * (wx + bias);
      if (z <= 0)
        nerr += 1;
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        cost += loss(z);
    }
  int n = imax - imin + 1;
  double wnorm =  dot(w,w) * wscale * wscale;
  cost = cost / n + 0.5 * lambda * wnorm;
  cout << prefix << setprecision(4)
       << "Misclassification: " << (double)nerr * 100.0 / n << "%." << endl;
  cout << prefix << setprecision(12) 
       << "Cost: " << cost << "." << endl;
}




// --- options

string trainfile;
string testfile;
double lambda = 1e-4;
int epochs = 5;
int maxtrain = -1;

void 
usage()
{
  cerr << "Usage: svmsgd [options] trainfile [testfile]" << endl
       << "Options:" << endl
       << " -lambda <lambda>" << endl
       << " -epochs <epochs>" << endl
       << " -maxtrain <n>" << endl
       << endl;
  exit(10);
}

void 
parse(int argc, const char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
        {
          if (trainfile.empty())
            trainfile = arg;
          else if (testfile.empty())
            testfile = arg;
          else
            usage();
        }
      else
        {
          while (arg[0] == '-') arg += 1;
          string opt = arg;
          if (opt == "lambda" && i+1<argc)
            {
              lambda = atof(argv[++i]);
              cout << "Using lambda=" << lambda << "." << endl;
              assert(lambda>0 && lambda<1e4);
            }
          else if (opt == "epochs" && i+1<argc)
            {
              epochs = atoi(argv[++i]);
              cout << "Going for " << epochs << " epochs." << endl;
              assert(epochs>0 && epochs<1e6);
            }
          else if (opt == "maxtrain" && i+1<argc)
            {
              maxtrain = atoi(argv[++i]);
              assert(maxtrain > 0);
            }
          else
            usage();
        }
    }
  if (trainfile.empty())
    usage();
}


// --- loading data

int dim;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

void
load(const char *fname, xvec_t &xp, yvec_t &yp)
{
  cout << "Loading " << fname << "." << endl;
  
  igzstream f;
  f.open(fname);
  if (! f.good())
    {
      cerr << "ERROR: cannot open " << fname << "." << endl;
      exit(10);
    }
  int pcount = 0;
  int ncount = 0;

  bool binary;
  string suffix = fname;
  if (suffix.size() >= 7)
    suffix = suffix.substr(suffix.size() - 7);
  if (suffix == ".dat.gz")
    binary = false;
  else if (suffix == ".bin.gz")
    binary = true;
  else
    {
      cerr << "ERROR: filename should end with .bin.gz or .dat.gz" << endl;
      exit(10);
    }

  while (f.good())
    {
      SVector x;
      double y;
      if (binary)
        {
          y = (f.get()) ? +1 : -1;
          x.load(f);
        }
      else
        {
          f >> y >> x;
        }
      if (f.good())
        {
          assert(y == +1 || y == -1);
          xp.push_back(x);
          yp.push_back(y);
          if (y > 0)
            pcount += 1;
          else
            ncount += 1;
          if (x.size() > dim)
            dim = x.size();
        }
    }
  cout << "Read " << pcount << "+" << ncount 
       << "=" << pcount + ncount << " examples." << endl;
}



int 
main(int argc, const char **argv)
{
  parse(argc, argv);

  // load training set
  load(trainfile.c_str(), xtrain, ytrain);
  cout << "Number of features " << dim << "." << endl;
  int imin = 0;
  int imax = xtrain.size() - 1;
  if (maxtrain > 0 && imax >= maxtrain)
    imax = imin + maxtrain -1;
  // prepare svm
  SvmSgd svm(dim, lambda);
  Timer timer;

  // load testing set
  if (! testfile.empty())
    load(testfile.c_str(), xtest, ytest);
  int tmin = 0;
  int tmax = xtest.size() - 1;

  for(int i=0; i<epochs; i++)
    {
      cout << "--------- Epoch " << i+1 << "." << endl;
      timer.start();
      svm.train(imin, imax, xtrain, ytrain);
      timer.stop();
      cout << "Total training time " << setprecision(6) 
           << timer.elapsed() << " secs." << endl;
      svm.test(imin, imax, xtrain, ytrain, "train: ");
      if (tmax >= tmin)
        svm.test(tmin, tmax, xtest, ytest, "test:  ");
    }
}
