function result = apply_xform(c, xform)
%% Apply the current transformation (3x3 matrix) to the current
%% coarse box c to get a new box g
%% if input is a matrix of bb's, then apply xform to all bbs
%% Tomasz Malisiewicz (tomasz@cmu.edu)

if size(c,1)==0
  result = c;
  return;
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
result = [d1 d2];

if CLIP == 1
  result = result(:,1:2);
end


% for i = 1:size(c,1)
%   curc = c(i,:);

%   xs(1:2,1) = curc([1 2])';
%   xs(1:2,2) = curc([3 4])';
%   xs(3,:) = 1;

%   d = xform*xs;
  
%   result(i,1) = d(1,1);
%   result(i,2) = d(2,1);
%   result(i,3) = d(1,2);
%   result(i,4) = d(2,2);  
% end
