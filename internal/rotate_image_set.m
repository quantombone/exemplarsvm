function sets = rotate_image_set(I)
%Return a set of images from a single one which applies different
%transformations to the input and is used for better matching

values1 = linspace(.5,2,10);
values2 = linspace(.5,2,10);
sets = cell(0,1);
for i = 1:length(values1)
  for j = 1:length(values2)
    r3 = eye(3)+randn(3,3);
    r3(:,end) = 0;
    r3(end) = 1;
    %tform = maketform('affine',r3);%[values2(j) 0 0; 0 values1(i) 0; 0 0 1]);
    tform = maketform('affine',[1 values2(j) 0; values1(i) 1 0; 0 0 1]);
    sets{end+1} = @()max(0.0,min(1.0,imtransform(toI(I),tform)));
  end
end