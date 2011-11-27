### ICCV2011 Abstract

This paper proposes a conceptually simple but surprisingly powerful method which combines the effectiveness of a discriminative object detector with the explicit correspondence offered by a nearest-neighbor approach. The method is based on training a separate linear SVM classifier for every exemplar in the training set. Each of these Exemplar-SVMs is thus defined by a single positive instance and millions of negatives. While each detector is quite specific to its exemplar, we empirically observe that an ensemble of such Exemplar-SVMs offers surprisingly good generalization. Our performance on the PASCAL VOC detection task is on par with the much more complex latent part-based model of Felzenszwalb et al., at only a modest computational cost increase. But the central benefit of our approach is that it creates an explicit association between each detection and a single training exemplar. Because most detections show good alignment to their associated exemplar, it is possible to transfer any available exemplar meta-data (segmentation, geometric structure, 3D model, etc.) directly onto the detections, which can then be used as part of overall scene understanding.

![](https://github.com/quantombone/exemplarsvm/raw/master/images/exemplar_classifiers-small_n.png)
---- 
**Author + Executive Exemplar-SVM Developer**: [Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/)
<br/>
**Exemplar-SVM Fellow Developer**: [Abhinav Shrivastava](http://www.abhinav-shrivastava.info/)
<br/>
**Exemplar-SVM Visionary**: [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg)
<br/>
**Exemplar-SVM Visionary**: [Alexei A. Efros](http://www.cs.cmu.edu/~efros)

###Please cite the following paper if you use this library:

Tomasz Malisiewicz, Abhinav Gupta, Alexei A. Efros. **Ensemble of Exemplar-SVMs for Object Detection and Beyond.** In ICCV, 2011. 
[PDF](http://www.cs.cmu.edu/~tmalisie/projects/iccv11/exemplarsvm-iccv11.pdf) 
[Project Page](http://www.cs.cmu.edu/~tmalisie/projects/iccv11/) 

###See my PhD thesis for more information and ICCV follow-up experiments:

Tomasz Malisiewicz. **Exemplar-based Representations for Object Detection, Association and Beyond.** PhD Dissertation, tech. report CMU-RI-TR-11-32. August, 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/thesis/malisiewicz_thesis.pdf)

###See our SIGGRAPH ASIA 2011 paper for image on image matching:
Abhinav Shrivastava, Tomasz Malisiewicz, Abhinav Gupta, Alexei A. Efros. **Data-driven Visual Similarity for Cross-domain Image Matching.** In SIGGRAPH ASIA, December 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/projects/sa11/shrivastava-sa11.pdf) [Project Page](http://graphics.cs.cmu.edu/projects/crossDomainMatching/) 

----
This object recognition library uses some great software:

* [libsvm-3.0-1](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

* fast blas convolution code (from [voc-release-4.0](http://www.cs.brown.edu/~pff/latent/)), 

* 31-D HOG feature code (from [voc-release-3.1](http://www.cs.brown.edu/~pff/latent/)), 

* [VOC development/evaluation code](http://pascallin.ecs.soton.ac.uk/challenges/VOC/) imported from the PASCAL VOC website

---
## Quickstart Guide

 * For training your own exemplars, see the notes in [exemplarsvm/demos/README.md](https://github.com/quantombone/exemplarsvm/blob/master/demos/README.md) and the main training script in [exemplarsvm/demos/voc_demo_esvm.m](https://github.com/quantombone/exemplarsvm/blob/master/demos/voc_demo_esvm.m)
 
 * For evaluating the PASCAL VOC 2007 pre-trained exemplars, see the notes in [exemplarsvm/demos/README.md](https://github.com/quantombone/exemplarsvm/blob/master/demos/README.md) and the main evaluation function in [exemplarsvm/demos/voc_demo_apply.m](https://github.com/quantombone/exemplarsvm/blob/master/demos/voc_demo_apply.m)

--- 
**Copyright (C) 2011 by Tomasz Malisiewicz**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

