function [data_set] = esvm_generate_dataset(Npos, Nneg, draw)
% Generate a synthetic dataset of circles (see esvm_demo_train_synthetic.m)
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if nargin == 0
  Npos = 3;
  Nneg = 10
elseif nargin == 1
  Nneg = Npos;
end

if ~exist('draw','var')
draw = 0;
end

%size of pattern to inpaint to the image
pattern_size = 200;

%size of real image
image_size = 400;

Apattern = generate_pattern(pattern_size);

N = Npos + Nneg;
data_set = cell(N,1);
fprintf(1,'Generating dataset of size %d posititives, %d negatives\n',...
        Npos, Nneg);
for i = 1:N
  
  %Generate a background image of goal size
  I = generate_random_image(image_size);
  %I = max(0.0,min(1.0,imresize(I,[image_size image_size], ...
  %                             'bicubic')));
  
  %rescale Apattern with random size, making sure it is resized
  %with nearest neighbor interpolation to ensure it remains binary
  random_scale = (rand*.8)+(1.0-.4);
  A = imresize(Apattern, random_scale, 'nearest');
   
  %If index i is greater than Npos, then it is a negative, so we
  %omit the repainting pattern
  if i <= Npos
    [I,bb] = inpaint_pattern(I,A);
  else
    bb = [];
  end
  
  [I,bb] = post_process(I,bb);  
  data_set{i}.I = I;
  
  if i <= Npos
    object.class = 'circle';
    object.view = '';
    object.truncated = 0;
    object.occluded = 0;
    object.difficult = 0;
    object.bbox = bb;
    object.polygon = [];    
    data_set{i}.objects = [object];
  end
  
  if draw
    figure(1)
    clf
    imagesc(I)
    plot_bbox(bb);
    title(sprintf('Image %d/%d',i,N));
    pause
  end
end

% Ineg = cell(Nneg,1);
% for i = 1:Nneg
%   I = rand(100,100,3);
%   I = rand(50,50,3);
%   I = max(0.0,min(1.0,imresize(I,[100 100],'bicubic')));
%   Ineg{i} = I;
% end

% data_set = cat(1,data_set,Ineg);

function Apattern = generate_pattern(pattern_size)
% Generate a pattern

A = zeros(pattern_size-1,pattern_size-1);
A(ceil(pattern_size/2),ceil(pattern_size/2)) = 1;
A = double(bwdist(A)<ceil(pattern_size*.4));
A = bwmorph(A,'remove');
A = bwmorph(A,'dilate',2);
[us,vs] = find(A);
A = A(min(us):max(us),min(vs):max(vs));
Apattern = repmat(A,[1 1 3]);

function I = generate_random_image(image_size)
%generate a random image, which can be load from disk too!
I = rand(image_size,image_size,3);

function [I,bb] = post_process(I,bb)
%perform additional post processing

I = .4*I+.6*rand(size(I));

function [I2,bb] = inpaint_pattern(I, A)
%find locations of where we can inpaint the pattern
sub1 = ceil(rand.*(size(I,1)-size(A,1)-1));
sub2 = ceil(rand.*(size(I,2)-size(A,2)-1));

Ipattern = zeros(size(I));
Ipattern(sub1+(1:size(A,1)),sub2+(1:size(A,2)),:) = A;
bb = [sub2 sub1 sub2+size(A,2) sub1+size(A,1) ];

% inds = find(Ipattern);
% I2 = I;
% I2(find(inds))=0;

I2 = (I.*(~Ipattern));



