function fire_rotate_image_set(I,model)
%Return a set of images from a single one which applies different
%transformations to the input and is used for better matching

I = toI(I);
msize = max(size(I));
I_best = I;
r_best = eye(3,3);
b_best = applyModel(I,model);
[alpha,beta] =max(b_best(:,end));
b_best = b_best(beta(1),:);
b_best(end)
for i = 1:100
  extra = eye(3)+randn(3,3)*.1;
  rnew = r_best + extra;
  rnew(3,3) = 1;
  rnew(1:2,end) = 0;
  tform = maketform('projective',rnew);
  curI = imtransform(I+randn(size(I))*.01,tform,'XYScale',1);
  curI = imresize_max(curI,msize*4);
  
  curI = I + randn(size(I))*.01;
  curI = max(0.0,min(1.0,curI));
  
  b = applyModel(curI,model);

  if numel(b) == 0
    b = [0 0 0 0 -1000];
  end
  
  [alpha,beta] = max(b(:,end));
  b = b(beta(1),:);
  b(1,end)

  
  if b(1,end) > b_best(1,end)
    b_best = b;
    I_best = curI;
    r_best = rnew;
  else
    fprintf(1,'.');
  end
  
  figure(1)
  clf
  subplot(1,2,1)
  imagesc(max(0.0,min(1.0,curI)))
  plot_bbox(b(1,:),num2str(b(1,end)))
  title('current box')
  subplot(1,2,2)
  imagesc(max(0.0,min(1.0,I_best)))
  plot_bbox(b_best(1,:),num2str(b_best(1,end)))
  title('Best box')
  
  drawnow

end