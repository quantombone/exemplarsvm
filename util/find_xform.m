function xform = find_xform(c, d)
%Finds the xform that maps from c to d
%in this case the xform is a simple translation and scaling
%xform is a [3 x 3] matrix which maps coordinates from c's frame to
%d's frame.  the transformation is applied to homogeneous
%coordinaes

%convert bounding box to cornners
xs(:,1) = c([1 2])';
xs(:,2) = c([3 2])';
xs(:,3) = c([1 4])';
xs(:,4) = c([3 4])';

ys(:,1) = d([1 2])';
ys(:,2) = d([3 2])';
ys(:,3) = d([1 4])';
ys(:,4) = d([3 4])';

xs(3,:) = 1;
ys(3,:) = 1;

A=ys*pinv(xs);
A(abs(A)<.000001) = 0;

xform = A;

