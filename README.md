---
ExemplarSVM readme
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

Tomasz Malisiewicz. **Exemplar-based Representations for Object Detection, Association and Beyond.** PhD Dissertation, tech. report CMU-RI-TR-11-32. August, 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/thesis/malisiewicz_thesis.pdf)

Abhinav Shrivastava, Tomasz Malisiewicz, Abhinav Gupta, Alexei A. Efros. **Data-driven Visual Similarity for Cross-domain Image Matching.** In SIGGRAPH ASIA, December 2011. [PDF](http://www.cs.cmu.edu/~tmalisie/projects/sa11/shrivastava-sa11.pdf) [Project Page](http://graphics.cs.cmu.edu/projects/crossDomainMatching/) 

----
This object recognition library is a combination of:
libsvm-3.0-1,
fast blas convolution code (from voc-release-4.0), 
31-D HOG feature code (from voc-release-3.1), 
Localization within a pyramid code (using both sliding windows and fast block-based method),
VOC development code imported from http://pascallin.ecs.soton.ac.uk/challenges/VOC/

---
Installation of PASCAL VOC Object Recognition Dataset
Go to http://pascallin.ecs.soton.ac.uk/challenges/VOC/ for more information

---
Copyright (C) 2011 by Tomasz Malisiewicz

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

