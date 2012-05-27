function [data_set] = esvm_generate_synthetic_dataset(Npos, Nneg, ...
                                                  cls, noise_factor, ...
                                                  bb_noise)
% Generate a synthetic dataset of circle patterns over a random
% background of noise (a very easy example) and assign the class
% name "cls" to those instances.  The first Npos images in the data
% will have one positive bounding box per image, and second Nneg
% images will not have any objects. Optional bb_noise [usually .05
% or .1] will add noise to the initial bounding placement which can
% be used to test how well the patterns get aligned.

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

%Default setting of bounding box noise is 0
if ~exist('bb_noise','var')
  bb_noise = 0;
end

if ~exist('noise_factor','var')
  noise_factor = .1;
end


%size of pattern to inpaint to the image
pattern_size = 200;

%size of real image
image_size = 400;

%NOTE: if the image is too small, then we will not be able to
% detect those instances
%Apattern = generate_pattern(pattern_size);
Apattern1 = generate_pattern(pattern_size,2);
Apattern1(:,1:round(size(Apattern1,2)/2),:)=[];
Apattern2 = Apattern1(:,end:-1:1,:);
%Apattern2 = generate_pattern(pattern_size,1);

N = Npos + Nneg;
data_set = cell(N,1);

fprintf(1,['Generating synthetic dataset of size: %d posititives,' ...
                                                  ' %d negatives\n'], ...
           Npos, Nneg);

for i = 1:N  
  

  if i <= Npos/2
    Apattern = Apattern1;
  else
    Apattern = Apattern2;
  end
  %Generate a background image of goal size
  I = generate_random_image(image_size,noise_factor);
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
  
  [I,bb] = post_process(I,bb,noise_factor);  
  data_set{i}.I = I;
  
  
  if i <= Npos
    object.class = cls;
    object.view = '';
    object.truncated = 0;
    object.occluded = 0;
    object.difficult = 0;
    object.bbox = bb;
    object.gtbox = bb;
    object.bbox = add_noise(object.bbox,bb_noise,I);
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


function Apattern = generate_pattern(pattern_size,index)
%Generate a random pattern

% for q = 1:10
%   patterns{q} = zeros(9,9);
% end

% patterns{1}(1,:) = 1;
% patterns{1}(end,:) = 1;
% patterns{1}(:,1) = 1;
% patterns{1}(:,end) = 1;

% patterns{2}(4:6,:) = 1;
% patterns{2}(:,4:6) = 1;
% Apattern = patterns{index};

% Apattern = imresize(Apattern,[pattern_size pattern_size], ...
%                     'nearest');
% Apattern = repmat(Apattern,[1 1 3]);
% return;

if ~exist('index','var') || index == 1
  % Generate a circular pattern of size pattern_size
  A = zeros(pattern_size-1,pattern_size-1);
  A(ceil(pattern_size/2),ceil(pattern_size/2)) = 1;
  A = double(bwdist(A)<ceil(pattern_size*.3));
  A = bwmorph(A,'remove');
  A = bwmorph(A,'dilate',ceil(pattern_size*.04));
  [us,vs] = find(A);
  A = A(min(us):max(us),min(vs):max(vs));
  Apattern = repmat(A,[1 1 3]);
else
  Apattern = zeros(9,9);
  Apattern(4:6,:) = 1;
  Apattern(:,4:6) = 1;
  Apattern = imresize(Apattern,[pattern_size pattern_size], ...
                      'nearest');
  Apattern = repmat(Apattern,[1 1 3]);
end

function I = generate_random_image(image_size,noise_factor)
% Generate a random image
% Note, loading a random image from disk might be a better
% background pattern
I = 1-noise_factor*rand(image_size,image_size,3);

function [I,bb] = post_process(I,bb,noise_factor)
%perform additional post processing such as adding noise to the
%image to make the detection problem more difficult

I = I + noise_factor*randn(size(I));
I = max(0.0,min(1.0,I));
%I = .4*I+noise_factor*rand(size(I));

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

function bb = add_noise(bb,factor,I)
if factor == 0
  return;
end

W = bb(3)-bb(1)+1;
H = bb(4)-bb(2)+1;
bb = round(bb + randn(size(bb))*factor*(W+H)/2);
bb(1) = max(0, bb(1));
bb(2) = max(0, bb(2));
bb(3) = min(size(I,2), bb(3));
bb(4) = min(size(I,1), bb(4));
