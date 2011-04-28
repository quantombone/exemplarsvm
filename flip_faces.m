function f = flip_faces(f,sizeI)
%Flip faces given size of the image
f(:,1) = sizeI(2)-f(:,1);
