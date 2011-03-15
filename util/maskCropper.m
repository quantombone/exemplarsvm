function [Iwhite,lilI,lilmask] = maskCropper(I,mask)
%function [Iwhite,lilI,lilmask] = maskCropper(I,mask)
%Given an Image and a Binary Segmentation Mask create a small icon
%showing only the image content inside the segment
%Iwhite is the resulting icon with background colored white
%lilI is the rectangular subimage defined by extent of mask
%lilmask is the small mask
%Tomasz Malisiewicz (tomasz@cmu.edu)

sss = [size(I,1) size(I,2)];

[uu,vv] = find(mask);
minu = min(uu);
maxu = max(uu);
minv = min(vv);
maxv = max(vv);

lilmask = mask(minu:maxu,minv:maxv);
I = I(minu:maxu,minv:maxv,:);

Iwhite = (I.*repmat(lilmask,[1 1 3])) + repmat(~lilmask,[1 1 3]);
lilI = I;