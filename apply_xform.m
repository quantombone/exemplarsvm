function result = apply_xform(c, xform)
%% Apply the current transformation (3x3 matrix) to the current
%% coarse box c to get a new box g
%% if input is a matrix of bb's, then apply xform to all bbs
%% Tomasz Malisiewicz (tomasz@cmu.edu)

if size(c,1)==0
  result = c;
  return;
end
for i = 1:size(c,1)
  curc = c(i,:);
  xs(:,1) = curc([1 2])';
  xs(:,2) = curc([3 2])';
  xs(:,3) = curc([1 4])';
  xs(:,4) = curc([3 4])';

  xs(3,:) = 1;

  d = xform*xs;
  d(3,:) = [];
  
  result(i,1) = d(1,1);
  result(i,2) = d(2,1);
  result(i,3) = d(1,2);
  result(i,4) = d(2,3);
  
  %result(i,:) = c(i,:).*xform.s + xform.a;
end
%result = zeros(1,4);
%result([1 3]) = (c([1 3])-xform.xaoffset)*xform.scaler + ...
%    xform.xboffset;

%result([2 4]) = (c([2 4])-xform.yaoffset)*xform.scaler2 + ...
%    xform.yboffset;
