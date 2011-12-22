function I = flip_image(I)
%Flips the image in the LR direction
for i = 1:size(I,3)
  I(:,:,i) = fliplr(I(:,:,i));
end