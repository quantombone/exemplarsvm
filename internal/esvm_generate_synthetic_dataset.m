function [data_set] = esvm_generate_dataset(Npos, Nneg, cls)
% Generate a synthetic dataset of circle patterns over a random
% background of noise (a very easy example) and assign the class
% name 'cls' to those instances.  The first Npos images in the data
% will have one positive bounding box per image, and second Nneg
% images will not have any objects.

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if nargin == 0
  Npos = 3;
  Nneg = 10
  cls = 'circle';
elseif nargin == 1
  Nneg = Npos;
  cls = 'circle'
end

%size of pattern to inpaint to the image
pattern_size = 200;

%size of real image
image_size = 400;

%NOTE: if the image is too small, then we will not be able to
% detect those instances
Apattern = generate_pattern(pattern_size);

N = Npos + Nneg;
data_set = cell(N,1);
fprintf(1,['Generating synthetic dataset of size: %d posititives,' ...
                                                  ' %d negatives\n'], ...
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
    object.class = cls;
    object.view = '';
    object.truncated = 0;
    object.occluded = 0;
    object.difficult = 0;
    object.bbox = bb;
    object.polygon = [];    
    data_set{i}.objects = [object];
  end
end

figure(1)
clf
vis_box_tiles(data_set);
title('Synthetic Data Set','FontSize',20)
axis image
axis off

function Apattern = generate_pattern(pattern_size)
% Generate a circular pattern of size pattern_size
A = zeros(pattern_size-1,pattern_size-1);
A(ceil(pattern_size/2),ceil(pattern_size/2)) = 1;
A = double(bwdist(A)<ceil(pattern_size*.4));
A = bwmorph(A,'remove');
A = bwmorph(A,'dilate',ceil(pattern_size*.04));
[us,vs] = find(A);
A = A(min(us):max(us),min(vs):max(vs));
Apattern = repmat(A,[1 1 3]);

function I = generate_random_image(image_size)
% Generate a random image
% Note, loading a random image from disk might be a better
% background pattern
I = rand(image_size,image_size,3);

function [I,bb] = post_process(I,bb)
%perform additional post processing such as adding noise to the
%image to make the detection problem more difficult

I = .4*I+.6*rand(size(I));

function [I2,bb] = inpaint_pattern(I, A)
%Inpaint pattern A into image I at some random location where the
%pattern fully fits

%find locations of where we can inpaint the pattern
sub1 = ceil(rand.*(size(I,1)-size(A,1)-1));
sub2 = ceil(rand.*(size(I,2)-size(A,2)-1));

Ipattern = zeros(size(I));
Ipattern(sub1+(1:size(A,1)),sub2+(1:size(A,2)),:) = A;
bb = [sub2 sub1 sub2+size(A,2) sub1+size(A,1) ];

I2 = (I.*(~Ipattern));
