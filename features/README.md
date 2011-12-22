Here are the features used in the Exemplar-SVM framework.

Compile with **features_compile.m** script

** features.m: a Matlab function which calles one of the mexed functions. It returns the dimensionality of the per-cell features when called without arguments.  The idea is that you can add your own features and 31 is never hard-coded in the codebase.

** features_pedro.cc: the 31-D features from voc-release-3.1

** features_raw.cc: the same features as features_pedro.cc but without the contrast normalization.  They are definitely worse than when using normalization, but the effect has not been fully studied.

** fconvblas.cc: a fast convolution procedure from from [voc-release-4](http://people.cs.uchicago.edu/~pff/latent-release4/)
