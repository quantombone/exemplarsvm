function [I,H2,W2] = rotate_pad(I,frac)
%Pad the image by rotating out the interior to the exterior part
%using padarray

H2 = 0;
W2 = 0;
return;

if ~exist('frac','var')
  frac = .2;
end
H = size(I,1);
W = size(I,2);

H2 = round(H*(frac));
W2 = round(W*(frac));

I2 = I(:,:,1)*0+1;
I = padarray(I,[H2 W2],0,'symmetric');

I2 = padarray(I2,[H2 W2],0);
b = bwdist(I2);
b1 = b(round(size(I2,1)/2),1);
b = b - min(b(:));
b = min(b,b1);
b = b / b1;% max(b(:));

%b = b.^2;
%b = b - min(b(:));
%b = b / max(b(:));
w = 1-b;
%I = I.*repmat(w,[1 1 3]);

%f = esvm_features(toI(I),8);
%I = jettify(HOGpicture(f - mean(f(:))));


