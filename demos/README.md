Inside this directory you will find two demo files, one to show off Exemplar-svm training, and the other to show how to apply a pre-trained ensemble of exemplars on a test-set of images. These demos are meant to work with the PASCAL VOC 2007 dataset, so let's download some data!

## Installing PASCAL VOC 2007 trainval/test sets

    mkdir /nfs/baikal/tmalisie/pascal
    cd /nfs/baikal/tmalisie/pascal
    wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar 
    tar xf VOCtrainval_06-Nov-2007.tar 
    
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

* Set working directory inside load_data_directory.m

* Run voc_demo_esvm (trains all bus exemplars).  To run this on a cluster, please see my [warp_scripts github repository](https://github.com/quantombone/warp_scripts)

