Welcome to the Exemplar-SVM library, a large-scale object recognition
library developed at Carnegie Mellon University while obtaining my PhD
in Robotics. 
  -- Tomasz Malisiewicz

The code is written in Matlab and is the basis of the following two
projects, as well as my doctoral dissertation:

## [Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg), [Alexei A. Efros](http://www.cs.cmu.edu/~efros). **Ensemble of Exemplar-SVMs for Object Detection and Beyond.** In ICCV, 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/projects/iccv11/exemplarsvm-iccv11.pdf) | [Project Page](http://www.cs.cmu.edu/~tmalisie/projects/iccv11/) 

![](https://github.com/quantombone/exemplarsvm/raw/master/images/exemplar_classifiers-small_n.png)

Abstract

This paper proposes a conceptually simple but surprisingly powerful method which combines the effectiveness of a discriminative object detector with the explicit correspondence offered by a nearest-neighbor approach. The method is based on training a separate linear SVM classifier for every exemplar in the training set. Each of these Exemplar-SVMs is thus defined by a single positive instance and millions of negatives. While each detector is quite specific to its exemplar, we empirically observe that an ensemble of such Exemplar-SVMs offers surprisingly good generalization. Our performance on the PASCAL VOC detection task is on par with the much more complex latent part-based model of Felzenszwalb et al., at only a modest computational cost increase. But the central benefit of our approach is that it creates an explicit association between each detection and a single training exemplar. Because most detections show good alignment to their associated exemplar, it is possible to transfer any available exemplar meta-data (segmentation, geometric structure, 3D model, etc.) directly onto the detections, which can then be used as part of overall scene understanding.

---

## [Abhinav Shrivastava](http://www.abhinav-shrivastava.info/), [Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg), [Alexei A. Efros](http://www.cs.cmu.edu/~efros). **Data-driven Visual Similarity for Cross-domain Image Matching.** In SIGGRAPH ASIA, December 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/projects/sa11/shrivastava-sa11.pdf) | [Project Page](http://graphics.cs.cmu.edu/projects/crossDomainMatching/)

![](https://github.com/quantombone/exemplarsvm/raw/master/images/sa_teaser.png)

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

More details and experimental evaluation can be found in my PhD thesis, available to download as a PDF.

[Tomasz Malisiewicz](http://www.cs.cmu.edu/~tmalisie/). **Exemplar-based Representations for Object Detection, Association and Beyond.** PhD Dissertation, tech. report CMU-RI-TR-11-32. August, 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/thesis/malisiewicz_thesis.pdf)

---- 

This object recognition library uses some great open-source software:

* Linear SVM training: [libsvm-3.0-1](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

* Fast blas convolution code (from [voc-release-4.0](http://www.cs.brown.edu/~pff/latent/)), 

* HOG feature code (31-D) (from [voc-release-3.1](http://www.cs.brown.edu/~pff/latent/)), 

* [VOC development/evaluation code](http://pascallin.ecs.soton.ac.uk/challenges/VOC/) imported from the PASCAL VOC website


----

# MATLAB Quick Start Guide

To get started, you need to install MATLAB and download the code from Github. This code has been tested on Mac OS X and Linux.  Pre-compiled Mex files for Mac OS X and Linux are included.

## Download Exemplar-SVM Library source code (MATLAB and C++) and compile it
``` sh
$ cd ~/projects/
$ git clone git@github.com:quantombone/exemplarsvm.git
$ cd ~/projects/exemplarsvm
$ matlab
>> esvm_compile
```

## Download and load pre-trained VOC2007 model(s)
``` sh
$ matlab
>> addpath(genpath(pwd))
>> [models, M, test_set] = esvm_download_models('voc2007-bus');
```

or

``` sh
$ wget http://people.csail.mit.edu/~tomasz/exemplarsvm/models/voc2007-models.tar
$ tar -xf voc2007-models.tar
$ matlab
>> load voc2007_bus.mat
>> [models, M, test_set] = esvm_download_models('voc2007-bus.mat');
```

You can alternatively download the pre-trained models individually from [http://people.csail.mit.edu/tomasz/exemplarsvm/models/](http://people.csail.mit.edu/tomasz/exemplarsvm/models/) or a tar file of all models [voc2007-models.tar](http://people.csail.mit.edu/tomasz/exemplarsvm/models/voc2007-models.tar) (NOTE: tar file is 450MB)

## Demo: Example of applying models to a single image or a set of images

See the demo walk-through [tutorial/esvm_demo_apply.html](http://people.csail.mit.edu/tomasz/exemplarsvm/tutorial/esvm_demo_apply.html) for a step-by-step tutorial on applying Exemplar-SVMs to images.
Or you can just run the demo:

``` sh
>> esvm_demo_apply;
```

# Training an Ensemble of Exemplar-SVMs

## Toy Demo: Exemplar-SVM training and testing on a set of synthetic images

See the synthetic training demo walk-through [tutorial/esvm_demo_train_synthetic.html](http://people.csail.mit.edu/tomasz/exemplarsvm/tutorial/esvm_demo_train_synthetic.html) for a step-by-step tutorial on how to set-up images and bounding boxes for a training experiment.
Or you can run the synthetic training demo:

``` sh
>> esvm_demo_train_synthetic;
```

The training scripts are designed to work with the PASCAL VOC 2007
dataset, so we need to download that first.

## Prerequsite: Install PASCAL VOC 2007 trainval/test sets
``` sh
$ mkdir /nfs/baikal/tmalisie/pascal #Make a directory for the PASCAL VOC data
$ cd /nfs/baikal/tmalisie/pascal
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar
$ tar xf VOCtest_06-Nov-2007.tar 
$ tar xf VOCtrainval_06-Nov-2007.tar 
``` 
You can also get the VOC 2007 dataset tar files manually, [VOCtrainval_06-Nov-2007.tar](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and [VOCtest_06-Nov-2007.tar](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar)

## Demo: Training and Evaluating an Ensemble of "bus" Exemplar-SVMs quick-demo
``` sh
>> data_dir = '/your/directory/to/pascal/VOCdevkit/';
>> dataset = 'VOC2007';
>> results_dir = '/your/results/directory/';
>> [models,M] = esvm_demo_train_voc_class_fast('car', data_dir, dataset, results_dir);
# All output (models, M-matrix, AP curve) has been written to results_dir
```

See the file [tutorial/esvm_demo_train_voc_class_fast.html](http://people.csail.mit.edu/tomasz/exemplarsvm/tutorial/esvm_demo_train_voc_class_fast.html) for a step-by-step tutorial on what esvm_demo_train_voc_class_fast.m produces


## Script: Training and Evaluating an Ensemble of "bus" Exemplar-SVMs full script
``` sh
>> data_dir = '/your/directory/to/pascal/VOCdevkit/';
>> dataset = 'VOC2007';
>> results_dir = '/your/results/directory/';
>> [models,M] = esvm_script_train_voc_class('bus', data_dir, dataset, results_dir);
# All output (models, M-matrix, AP curve) has been written to results_dir
```

# Extra: How to run the Exemplar-SVM framework on a cluster

This library was meant to run on a cluster with a shared NFS/AFS file
structure where all nodes can read/write data from a common data
source/target.  The PASCAL VOC dataset must be installed on such a
shared resource and the results directory as well.  The idea is that
results are written as .mat files and intermediate work is protected
via lock files. Lock files are temporary files (they are directories
actually) which are deleted once something has finished process.  This
means that the entire voc training script can be replicated across a
cluster, you can run the script 200x times and the training will
happen in parallel.

To run ExemplarSVM on a cluster, first make sure you have a cluster,
use an ssh-based launcher such as my
[warp_scripts](https://github.com/quantombone/warp_scripts) github
project.  I have used warp_starter.sh at CMU (using WARP cluster)
and sc.sh at MIT (using the continents).

### Here is the command I often use at MIT to start Exemplar-SVM runs, where machine_list.sh contains computer names
``` sh
$ cd ~
$ git clone git@github.com:quantombone/warp_scripts.git
$ cd ~/warp_scripts/
$ cp machine_list.sh-example machine_list.sh
$ nano machine_list.sh #now edit the file to point to your cluster CPUs
$ ./sc.sh "cd ~/projects/exemplarsvm; addpath(genpath(pwd)); esvm_script_train_voc_class('train');"
```

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

