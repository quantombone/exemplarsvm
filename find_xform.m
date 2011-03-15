function xform = find_xform(c, d)
%finds the xform that maps from c to d
%in this case the xform is a simple translation and scaling
%xform is a [3 x 3] matrix which maps coordinates from c's frame to
%d's frame.  the transformation is applied to homogeneous coordinaes

% aoffset = [c(1) d(1)];
% scaler1 = (d(3) - aoffset(2)) / (c(3) - aoffset(1));

% boffset = [c(2) d(2)];
% scaler2 = (d(4) - boffset(2)) / (c(4) - boffset(1));

% xform.s = [scaler1 scaler2 scaler1 scaler2];
% xform.a = [boffset(1) boffset(2) boffset(1) boffset(2)] - ...
%           [aoffset(1) aoffset(2) aoffset(1) ...
%            aoffset(2)].*xform.s;

% norm(apply_xform(c,xform)-d)
% keyboard

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

% return;
% f2 = xs;
% f2(3,:) = 1;

% f2 = A*f2;
% f2(3,:) = [];
% f = f2';



% xa = c([1 3]);
% xb = d([1 3]);

% xaoffset = xa(1);
% xa = xa - xaoffset;

% xboffset = xb(1);
% xb = xb - xboffset;

% scaler = (xb(2)/xa(2));

% ya = c([2 4]);
% yb = d([2 4]);

% yaoffset = ya(1);
% ya = ya - yaoffset;

% yboffset = yb(1);
% yb = yb - yboffset;

% scaler2 = (yb(2)/ya(2));



% %xform.xaoffset = xaoffset;
% %xform.xboffset = xboffset;
% %xform.yaoffset = yaoffset;
% %xform.yboffset = yboffset;
% %xform.scaler = scaler;
% %xform.scaler2 = scaler2;


% %d1 = (c - [xaoffset yaoffset xaoffset yaoffset]).*[scaler scaler2 scaler ...
% %                    scaler2] + [xboffset yboffset xboffset yboffset];

% %d2 = (c .*[scaler scaler2 scaler scaler2]) - [xaoffset yaoffset xaoffset ...
% %                    yaoffset].*[scaler scaler2 scaler scaler2] + ...
% %     [xboffset yboffset xboffset yboffset];

% xform.s = [scaler scaler2 scaler scaler2];
% xform.a = [xboffset yboffset xboffset yboffset] - ...
%           [xaoffset yaoffset xaoffset ...
%            yaoffset].*[scaler scaler2 scaler scaler2];

% %norm(apply_xform(c,xform)-d)
% %keyboard

% %