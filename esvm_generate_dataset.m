function [Is,bbs,Ineg] = esvm_generate_dataset(N)

Npos = N;
Nneg = N;
  

A = zeros(39,39);
A(20,20)=1;
A = double(bwdist(A)<15);
A = bwmorph(A,'remove');
A = bwmorph(A,'dilate',2);
A = repmat(A,[1 1 3]);

for i = 1:Npos
  I = rand(100,100,3);
  I = rand(50,50,3);
  I = max(0.0,min(1.0,imresize(I,[100 100],'bicubic')));
  sub1 = ceil(rand.*(size(I,1)-size(A,1)-1));
  sub2 = ceil(rand.*(size(I,2)-size(A,2)-1));
  %A2 = A + rand(size(A)).*(A<.9);
  I2 = zeros(size(I));
  I2(sub1+(1:size(A,1)),sub2+(1:size(A,2)),:)=A;

  inds = find(I2);
  I(inds) = 0;
  
  Is{i} = I;
  bbs{i} = [sub2 sub1 sub2+size(A,2) sub1+size(A,1) ];
  
  if 0
  figure(1)
  clf
  imagesc(I)
  plot_bbox(bbs{i});

  pause
  end
end

for i = 1:Nneg
  I = rand(100,100,3);
  I = rand(50,50,3);
  I = max(0.0,min(1.0,imresize(I,[100 100],'bicubic')));
  Ineg{i} = I;
end