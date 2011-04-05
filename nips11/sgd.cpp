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

