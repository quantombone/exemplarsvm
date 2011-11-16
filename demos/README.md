Here are the Exemplar-SVM main functions for starting the training procedure.
These demos are meant to work with the PASCAL VOC 2007 dataset.

# Running on a cluster information This library was meant to run on a
cluster with a shared NFS/AFS file structure where all nodes can
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

