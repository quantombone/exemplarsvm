Inside this directory you will find two demo files, one to show off Exemplar-svm training, and the other to show how to apply a pre-trained ensemble of exemplars on a test-set of images. These demos are meant to work with the PASCAL VOC 2007 dataset, so let's download some data!

## Installing PASCAL VOC 2007 trainval/test sets
``` sh
$ mkdir /nfs/baikal/tmalisie/pascal #Make a directory for the PASCAL VOC data
$ cd /nfs/baikal/tmalisie/pascal
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar
$ tar xf VOCtest_06-Nov-2007.tar 
$ tar xf VOCtrainval_06-Nov-2007.tar 
``` 

## Downloading Exemplar-SVM MATLAB source code
``` sh
$ cd ~/projects/
$ git clone git@github.com:quantombone/exemplarsvm.git #this will make ~/projects/exemplarsvm the code directory
```

## Setup paths

This quickstart guide assumes you have hardcoded "/nfs/baikal/tmalisie/pascal/VOCdevkit/" into the function load_data_directory.m, as well as the results folder "/nfs/baikal/tmalisie/esvm-data/" into "load_results_directory.m"

## Making sure Exemplar-SVM library is compiled and working
``` sh
$ cd ~/projects/exemplarsvm
$ matlab
$ >> addpath(genpath(pwd));
$ >> cd features/
$ >> compile;
```

## Set data
---
At this point you can choose to either train your own detector, or download the pre-trained PASCAL VOC 2007 detectors and use them to create detections in a collection of images
---
# Train, Calibrate, and Test pipeline

## Training an Exemplar-SVM "bus" detector
``` sh
$ cd ~/projects/exemplarsvm
$ matlab
$ >> addpath(genpath(pwd));
$ >> voc_demo_esvm;
```

# Applying pre-trained models to a set of images
## Download pre-trained models into results direcotry
``` sh
$ cd /nfs/baikal/tmalisie/esvm-data/
$ wget http://people.csail.mit.edu/tomasz/exemplarsvm-VOC2007-models.tar.gz
$ gzip -d exemplarsvm-VOC2007-models.tar.gz
$ tar xf exemplarsvm-VOC2007-models.tar #now the results directory contains all 20 category PASCAL VOC 2007 models
```
## How to run the Exemplar-SVM framework on a cluster 

This library was meant to run on a cluster with a shared NFS/AFS file structure where all nodes can
read/write data from a common data source.  The PASCAL VOC dataset
must be installed there and the scratch space must also be present
there.  The idea is that scratch-work is written as .mat files and
intermediate work is protected via .lock files.  lock files are
temporary files (they are directories actually) which are deleted once
something has finished process.  This means that the entire script can
be replicated across a cluster, you can run the script 200x times and
the training will happen in parallel.


* Install PASCAL VOC dataset

* Run voc_demo_esvm (trains all bus exemplars).  To run this on a cluster, please see my [warp_scripts github repository](https://github.com/quantombone/warp_scripts)

