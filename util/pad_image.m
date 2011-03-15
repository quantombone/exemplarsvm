function I2 = pad_image(I, p, val)
if ~exist('val','var')
  val = 0;
end
%pad an image by this amount
I2 = ones(size(I,1)+p*2,size(I,2)+p*2,size(I,3))*val;
if (p>0)
  I2(p+(1:size(I,1)),p+(1:size(I,2)),:) = I;
else
  I2=I(-p+1:(size(I,1)+p),-p+1:(size(I,2)+p),:);
end
  