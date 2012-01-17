function [Ipos,Ineg] = esvm_generate_dataset(Npos,Nneg)  

if nargin == 0
  Npos = 3;
  Nneg = 10
elseif nargin == 1
  Nneg = 10;
end

A = zeros(39,39);
A(20,20)=1;
A = double(bwdist(A)<15);
A = bwmorph(A,'remove');
A = bwmorph(A,'dilate',2);
Asave = repmat(A,[1 1 3]);

Ipos = cell(Npos,1);
for i = 1:Npos
  I = rand(100,100,3);
  I = rand(50,50,3);
  rscale = (rand*.8)+(1.0-.4);
  A = imresize(Asave,rscale,'nearest');

  I = max(0.0,min(1.0,imresize(I,[100 100],'bicubic')));
  sub1 = ceil(rand.*(size(I,1)-size(A,1)-1));
  sub2 = ceil(rand.*(size(I,2)-size(A,2)-1));
  %A2 = A + rand(size(A)).*(A<.9);
  I2 = zeros(size(I));
  I2(sub1+(1:size(A,1)),sub2+(1:size(A,2)),:)=A;

  inds = find(I2);
  I(inds) = 0;
  
  Irand = rand(size(I));
  I = .8*I+.2*Irand;
  
  Ipos{i}.I = I;
  
  recs.folder = '';
  recs.filename = '';
  recs.source = '';
  [recs.size.width,recs.size.height,recs.size.depth] = size(I);
  recs.segmented = 0;
  recs.imgname = sprintf('%08d',i);;
  recs.imgsize = size(I);
  recs.database = '';

  object.class = 'circle';
  object.view = '';
  object.truncated = 0;
  object.occluded = 0;
  object.difficult = 0;
  object.label = 'circle';
  object.bbox = [sub2 sub1 sub2+size(A,2) sub1+size(A,1) ];
  object.bndbox.xmin =object.bbox(1);
  object.bndbox.ymin =object.bbox(2);
  object.bndbox.xmax =object.bbox(3);
  object.bndbox.ymax =object.bbox(4);
  object.polygon = [];
  recs.objects = [object];
  %object.mask = [];
  %object.hasparts = 0;
  %object.par
  
  Ipos{i}.recs = recs;
  % Ipos{i}.bbox = 
  % Ipos{i}.cls = 'synthetic';
  % Ipos{i}.curid = sprintf('%05d',i);
  % filer = sprintf('%s.%d.%s.mat', Ipos{i}.curid, 1, ...
  %                 'synthetic');
  % Ipos{i}.filer = filer;
  % Ipos{i}.objectid = 1;
  if 0
  figure(1)
  clf
  imagesc(I)
  plot_bbox(bbs{i});

  pause
  end
end

Ineg = cell(Nneg,1);
for i = 1:Nneg
  I = rand(100,100,3);
  I = rand(50,50,3);
  I = max(0.0,min(1.0,imresize(I,[100 100],'bicubic')));
  Ineg{i} = I;
end