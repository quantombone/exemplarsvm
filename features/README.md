Here are the features used in the Exemplar-SVM framework.

features.m is a Matlab function which calles one of the mexed functions.

features_pedro.cc are the 31-D features from voc-release-3.1

http://people.cs.uchicago.edu/~pff/latent-release3/

P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan. Object Detection with Discriminatively Trained Part Based Models. PAMI 2010.

features_raw.cc are the same features but without the contrast normalization.  They are definitely worse than when using normalization, but the effect has not been fully studied.