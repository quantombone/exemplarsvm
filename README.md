Welcome to the Exemplar-SVM library.  The code is written in Matlab
and is the basis of the following two projects:

[Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg), [Alexei A. Efros](http://www.cs.cmu.edu/~efros). **Ensemble of Exemplar-SVMs for Object Detection and Beyond.** In ICCV, 2011. 
[PDF](http://www.cs.cmu.edu/~tmalisie/projects/iccv11/exemplarsvm-iccv11.pdf) 
[Project Page](http://www.cs.cmu.edu/~tmalisie/projects/iccv11/) 

![](https://github.com/quantombone/exemplarsvm/raw/master/images/exemplar_classifiers-small_n.png)

Abstract

This paper proposes a conceptually simple but surprisingly powerful method which combines the effectiveness of a discriminative object detector with the explicit correspondence offered by a nearest-neighbor approach. The method is based on training a separate linear SVM classifier for every exemplar in the training set. Each of these Exemplar-SVMs is thus defined by a single positive instance and millions of negatives. While each detector is quite specific to its exemplar, we empirically observe that an ensemble of such Exemplar-SVMs offers surprisingly good generalization. Our performance on the PASCAL VOC detection task is on par with the much more complex latent part-based model of Felzenszwalb et al., at only a modest computational cost increase. But the central benefit of our approach is that it creates an explicit association between each detection and a single training exemplar. Because most detections show good alignment to their associated exemplar, it is possible to transfer any available exemplar meta-data (segmentation, geometric structure, 3D model, etc.) directly onto the detections, which can then be used as part of overall scene understanding.

---

[Abhinav Shrivastava](http://www.abhinav-shrivastava.info/), [Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg), [Alexei A. Efros](http://www.cs.cmu.edu/~efros). **Data-driven Visual Similarity for Cross-domain Image Matching.** In SIGGRAPH ASIA, December 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/projects/sa11/shrivastava-sa11.pdf) [Project Page](http://graphics.cs.cmu.edu/projects/crossDomainMatching/) 

![](https://github.com/quantombone/exemplarsvm/raw/v1/images/sa_teaser.png)

Abstract

The goal of this work is to find visually similar images even if they
appear quite different at the raw pixel level. This task is
particularly important for matching images across visual domains, such
as photos taken over different seasons or lighting conditions,
paintings, hand-drawn sketches, etc. We propose a surprisingly simple
method that estimates the relative importance of different features in
a query image based on the notion of "data-driven uniqueness". We
employ standard tools from discriminative object detection in a novel
way, yielding a generic approach that does not depend on a particular
image representation or a specific visual domain. Our approach shows
good performance on a number of difficult cross-domain visual tasks
e.g., matching paintings or sketches to real photographs. The method
also allows us to demonstrate novel applications such as Internet
re-photography, and painting2gps.

---

[Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/). **Exemplar-based Representations for Object Detection, Association and Beyond.** PhD Dissertation, tech. report CMU-RI-TR-11-32. August, 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/thesis/malisiewicz_thesis.pdf)

---- 


This object recognition library uses some great open-source software:

* Linear SVM training: [libsvm-3.0-1](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

* Fast blas convolution code (from [voc-release-4.0](http://www.cs.brown.edu/~pff/latent/)), 

* HOG feature code (31-D) (from [voc-release-3.1](http://www.cs.brown.edu/~pff/latent/)), 

* [VOC development/evaluation code](http://pascallin.ecs.soton.ac.uk/challenges/VOC/) imported from the PASCAL VOC website


----

# MATLAB Quick Start Guide

To get started, you need to install MATLAB and download the code from Github.

## Download Exemplar-SVM MATLAB source code
``` sh
cd ~/projects/
git clone git@github.com:quantombone/exemplarsvm.git
cd ~/projects/exemplarsvm
```

## Download and load pre-trained VOC2007 model(s)

``` sh
$ matlab
$ addpath(genpath(pwd))
$ >> esvm_download_models('bus');
$ >> load voc2007-bus.mat #vars "models" and "M" are loaded
```

``` sh
$ >> load bus_test_set.mat
$ >> esvm_demo_apply_exemplars(bus_set, models, M);
```

### or load your own image
``` sh
$ matlab
$ >> I = imread('image1.png'); #your own image
$ >> esvm_demo_apply_exemplars(I, models, M);
```

### or load your own set of images
``` sh
$ matlab
$ >> I1 = imread('image1.png'); #your own image
$ >> ...
$ >> IN = imread('imageN.png'); #your own image
$ >> Iarray = {I1, ..., IN};
$ >> esvm_demo_apply_exemplars(Iarray, models, M)
```

### a directory of images
``` sh
$ matlab
$ >> Idirectory = '~/images/';
$ >> esvm_demo_apply_exemplars(Idirectory, models, M)
```

---

###Also, you can download all models

``` sh
$ cd ~/projects/exemplarsvm/
$ wget http://people.csail.mit.edu/~tomasz/exemplarsvm/voc2007-models.tar
$ tar -xf voc2007-models.tar
```

then in MATLAB, you can load models by their name:

``` sh
$ matlab
$ >> load voc2007_bus.mat
```

### Make sure Exemplar-SVM library is compiled and working
``` sh
$ matlab
$ >> cd features/
$ >> features_compile;
$ >> cd ../util/
$ >> util_compile;
$ >> cd ../libsvm/
$ >> libsvm_compile;
```

---

# 1) Train and Test an Ensemble of Exemplar-SVMs from scratch

---

The training demos are meant to work with the PASCAL VOC 2007 dataset,
so let's download some data!

## Install PASCAL VOC 2007 trainval/test sets
``` sh
$ mkdir /nfs/baikal/tmalisie/pascal #Make a directory for the PASCAL VOC data
$ cd /nfs/baikal/tmalisie/pascal
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar
$ tar xf VOCtest_06-Nov-2007.tar 
$ tar xf VOCtrainval_06-Nov-2007.tar 
``` 

## Edit esvm_script_train_voc_class.m 
``` sh
data_directory = '/your/directory/to/pascal/VOCdevkit/';
results_directory = '/your/results/directory/';
```

## Training and Evaluating an Ensemble of "bus" Exemplar-SVMs
``` sh
$ matlab
$ addpath(genpath(pwd))
$ >> [models,M] = esvm_script_train_voc_class('bus');
```

# Extra: How to run the Exemplar-SVM framework on a cluster 

This library was meant to run on a cluster with a shared NFS/AFS file
structure where all nodes can read/write data from a common data
source.  The PASCAL VOC dataset must be installed there and the
scratch space must also be present there.  The idea is that
scratch-work is written as .mat files and intermediate work is
protected via .lock files.  lock files are temporary files (they are
directories actually) which are deleted once something has finished
process.  This means that the entire script can be replicated across a
cluster, you can run the script 200x times and the training will
happen in parallel.

To run ExemplarSVM on a cluster, first make sure you have a cluster,
then please see my
[warp_scripts](https://github.com/quantombone/warp_scripts) github
repository

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

