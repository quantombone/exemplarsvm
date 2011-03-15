function [normos,overlaps,side1,side2] = getosmatrix(feats1,feats2)
%%feats is a matrix of NPIXELS x NSEGMENTS 
%%and os is the overlap score matrix of size NSEGMENTS x NSEGMENTS
%%which computes the overlap score between each pair of segments

if ~exist('feats2','var')
  feats2 = feats1;
end

overlaps = feats1' * feats2;
sizers1 = sum(feats1,1);
sizers2 = sum(feats2,1);
siz1 = repmat(sizers1',1,size(feats2,2));
siz2 = transpose(repmat(sizers2',1,size(feats1,2)));

normos = overlaps ./ (siz1 + siz2 - overlaps);

side1 = overlaps ./ siz1;
side2 = overlaps ./ siz2;
%funkyos = max(overlaps ./siz1,overlaps./siz2);
