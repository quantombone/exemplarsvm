function [os,os1] = getosmatrix_bb(boxes,gts)
%% Given two sets of bounding boxes, N1 in the first one, and N2 in
%% the second one, compute a N1xN2 overlap score matrix where the
%% overlap score is the ratio of the intersection to union
%% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('gts','var')
  gts = boxes;
end

if numel(boxes) == 0 || numel(gts) == 0
  os = zeros(size(boxes,1),size(boxes,2));
  return;
end

x1 = boxes(:,1);
y1 = boxes(:,2);
x2 = boxes(:,3);
y2 = boxes(:,4);

area = (x2-x1+1) .* (y2-y1+1);

xa1 = gts(:,1);
ya1 = gts(:,2);
xa2 = gts(:,3);
ya2 = gts(:,4);

area2 = (xa2-xa1+1) .* (ya2-ya1+1);

os = zeros(size(boxes,1),size(gts,1));
if size(boxes,1)*size(gts,1)==0
  return;
end

if nargout == 2
  os1 = os;
end

for i = 1:size(boxes,1)
  
  xx1 = max(boxes(i,1),gts(:,1));
  yy1 = max(boxes(i,2),gts(:,2));
  xx2 = min(boxes(i,3),gts(:,3));
  yy2 = min(boxes(i,4),gts(:,4));

  w = xx2-xx1+1;
  h = yy2-yy1+1;
  
  o = w.*h;
  o( (w<0) | (h<0) ) = 0;

  os(i,:) = o ./ (area(i) + area2 - o);
  
  if nargout == 2
    os1(i,:) = o ./ area2;
    %os1(i,:) = o ./ area(i);
  end
end

  

