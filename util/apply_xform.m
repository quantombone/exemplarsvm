function transformed_c = apply_xform(c, xform)
% function transformed_c = apply_xform(c, xform)
% Apply the current transformation (3x3 matrix) to the current
% bounding box c to get a new bounding box transformed_c
% if input is a matrix of bb's, then apply xform to all bbs
% bb must have 4 numbers (or just 2 numbers to map a point)
% transformed_c will be same size as c, with all non-bb fields left
% intact

% Tomasz Malisiewicz (tomasz@cmu.edu)

transformed_c = c;

if size(c,1)==0
  return;
end

if size(c,2) == 1 || size(c,2) == 3
  error('apply_xform: invalid size if input');
end
  

CLIP = 0;
if size(c,2) == 2
  CLIP = 1;
  c(:,[3 4]) = 0;
end
  
xs = c(:,1:2)';
xs(3,:) = 1;
d1 = xform*xs;
d1 = d1(1:2,:)';

xs = c(:,3:4)';
xs(3,:) = 1;
d2 = xform*xs;
d2 = d2(1:2,:)';
maxd = 4;%min(4,size(transformed_c,2));

transformed_c(:,1:maxd) = [d1 d2];

if CLIP == 1
  transformed_c = transformed_c(:,1:2);
end


% for i = 1:size(c,1)
%   curc = c(i,:);

%   xs(1:2,1) = curc([1 2])';
%   xs(1:2,2) = curc([3 4])';
%   xs(3,:) = 1;

%   d = xform*xs;
  
%   transformed_c(i,1) = d(1,1);
%   transformed_c(i,2) = d(2,1);
%   transformed_c(i,3) = d(1,2);
%   transformed_c(i,4) = d(2,2);  
% end
