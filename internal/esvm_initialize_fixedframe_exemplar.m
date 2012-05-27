function model = esvm_initialize_fixedframe_exemplar(I, bbox, ...
                                                  params)
% Initialize exemplars using a Fixed Frame as defined by
% init_params.hg_size.  The fixedframe mode also needs to compute a
% template-mask, which will indicate which part of the square
% template is actually being used.  This mask will in general have
% zero regions because we enforce the mask to be square (via
% padding).
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

origbox = bbox;
init_params = params.init_params;
sbin = init_params.sbin;
hg_size = init_params.hg_size;
params.init_params = init_params;

%NOTE: K/ADD_LR are advanced features
%K/ADD_LR determines how many windows to get templates for (although only
%the first will have its template mask region defined)

if isfield(init_params,'K')
  K = init_params.K;
else
  K = 1;
end

%make sure K is at least 1
K = max(K, 1);


if isfield(init_params,'ADD_LR') && init_params.ADD_LR == 1 && (K >= 1)
  %fprintf(1,'ADD_LR turned on K = %d\n', K);
  %params = init_params;
  params.init_params = rmfield(params.init_params,'ADD_LR');
  model = esvm_initialize_fixedframe_exemplar(I, ...
                                      bbox, ...
                                      params);

  model2 = esvm_initialize_fixedframe_exemplar(flip_image(I), ...
                                      flip_box(bbox, size(I)), ...
                                      params);

  x2 = model2.x;
  bb2 = model2.bb;
  for j = 1:size(bb2,1)
    bb2(j,7) =  1;
    bb2(j,1:4) = flip_box(bb2(j,1:4),size(I));
  end
  model.x = cat(2,model.x,x2);
  model.bb = cat(1,model.bb,bb2);
  return;
end

rawbox = bbox;

%NOTE: is this even worth it?
bbox = slight_expand(bbox);
bbox = squareize_bbox(bbox);

%Compute pyramid 
clear t
params.detect_levels_per_octave = 10;
[t.hog, t.scales] = esvm_pyramid(I, params);
t.padder = 5;

%Pad pyramid and compute all bounding boxes, and their levels
[allbb,alluv,alllvl,t] = pad_and_get_all_bb(t, hg_size, sbin);

%Get all overlaps with fixed frame box (ideal is 1.0)
[os,os1] = getosmatrix_bb(allbb,bbox);

%% get the symmetry meaasure meansym which measures the symmetry
%distance (ideal would be zero)
sym1 = abs((allbb(:,1)-bbox(1)) + (allbb(:,3)-bbox(3)));
sym2 = abs((allbb(:,2)-bbox(2)) + (allbb(:,4)-bbox(4)));
meansym = (sym1+sym2)/(bbox(3)-bbox(1)+bbox(4)-bbox(2))/8;

%TJM: hack tufn off meansym
%meansym = 100;

%Sorty by a measure which wants high overlap, and low symmetry-difference
[tmp,order] = sort(os-meansym+.01*os1,'descend');

curfeats = cell(K,1);
bbs = cell(K,1);

for q = 1:K
  superind = order(q);

  curfeat = t.hog{alllvl(superind)}...
            (alluv(superind,1)-1+(1:hg_size(1)),...
             alluv(superind,2)-1+(1:hg_size(2)),:);

  level = alllvl(superind);
  ip.scale = t.scales(level);
  ip.offset = alluv(superind,:) - t.padder;
  ip.flip = 0;
  ip.bb = allbb(superind,:);
  
  bb = zeros(1,12);
  bb(1:4) = allbb(superind,:);
  bb(7) = ip.flip;
  bb(8) = ip.scale;
  bb(9:10) = ip.offset;
  bb(12) = 0;
  
  bbs{q} = bb;
  curfeats{q} = curfeat;
end


% Get the template mask, so that we only perform learning over the
% regions of the template corresponding to the GT region.

I2 = I*0;
rawbox = clip_to_image(slight_expand(rawbox),[1 1 size(I,2) size(I, ...
                                                  1)]);
rawbox = rawbox;

I2(rawbox(2):rawbox(4),rawbox(1):rawbox(3),:) = 1;
Ikeep = I2;
oks = find(I2(:));
I2(oks) = double(randn(size(oks))>0);

I2 = resize(I2,bbs{1}(8));
Ikeep = imresize(Ikeep,[size(I2,1) size(I2,2)],'nearest');
Icopy = resize(I,bbs{1}(8));


f2 = features_raw(I2,sbin);
f2 = padarray(f2,[t.padder+1 t.padder+1 0]);

%maxers=ceil(size(Ikeep)/sbin)*sbin
%Ikeep(maxers(1),maxers(2),3) = 0;
%f3 = imresize(Ikeep,[size(f2,1) size(f2,2)],'nearest');




f2 = f2(bbs{1}(9)+(1:hg_size(1))-1+t.padder,...
        bbs{1}(10)+(1:hg_size(2))-1+t.padder,:);


%f3 = f3(bbs{1}(9)+(1:hg_size(1))-1+t.padder,...
%        bbs{1}(10)+(1:hg_size(2))-1+t.padder,:);


fcopy = esvm_features(Icopy,sbin);
fcopy = padarray(fcopy,[t.padder+1 t.padder+1 0]);

fcopy = fcopy(bbs{1}(9)+(1:hg_size(1))-1+t.padder,...
              bbs{1}(10)+(1:hg_size(2))-1+t.padder,:);

fmask2 = (sum(f2>0,3)>=4);
fmask = sum(f2.^2,3)>0;


% figure(333)
% subplot(2,2,1)
% imagesc(I)
% plot_bbox(bbs{1})
% subplot(2,2,3)
% imagesc(fmask)
% subplot(2,2,4)
% imagesc(fmask2)
% subplot(2,2,2), imagesc(sum(f2.^2,3))
% pause

[u,v] = find(fmask);
fmask(min(u):max(u),min(v):max(v)) = 1;
x = curfeats{1};
model.mask = (sum(x.^2,3)>0) & fmask;

model.masks = cellfun(@(x,y)get_mask(I,rawbox,x,y,sbin,t,hg_size),bbs, ...
                curfeats,'UniformOutput',false);
model.masks = cat(3,model.masks{:});


model.init_params = init_params;
model.hg_size = [hg_size(1) hg_size(2) init_params.features()];





model.w = zeros(size(x));

mask3 = repmat(model.mask,[1 1 init_params.features()]);
mask3 = mask3(:);

%%initialize model to normalized HOG
model.w(mask3) = curfeats{1}(mask3) - mean(curfeats{1}(mask3));
model.b = 0;

model.x = cellfun2(@(x)reshape(x,[],1), curfeats);
model.x = cat(2,model.x{:});
model.bb = cat(1,bbs{:});
model.bb(:,end) = model.w(:)'*model.x;


function bbox = squareize_bbox(bbox)
%Expand region such that is still within image and tries to satisfy
%these constraints best
%requirements: each dimension is at least 50 pixels, and max aspect
%ratio os (.25,4)

w = bbox(3)-bbox(1)+1;
h = bbox(4)-bbox(2)+1;

if w < h
  neww = h;
  differ = neww - w;
  left = ceil(differ/2);
  right = floor(differ/2);
  bbox(3) = bbox(3) + right;
  bbox(1) = bbox(1) - left;
elseif h < w
  newh = w;
  differ = newh - h;
  left = ceil(differ/2);
  right = floor(differ/2);
  bbox(4) = bbox(4) + right;
  bbox(2) = bbox(2) - left;
end


function bbox = slight_expand(bbox,fraction)
if ~exist('fraction','var')
  fraction = 0;
end
offset1 = (bbox(3)-bbox(1))*fraction;
offset2 = (bbox(4)-bbox(2))*fraction;
bbox(1) = bbox(1) - offset1;
bbox(3) = bbox(3) + offset1;

bbox(2) = bbox(2) - offset2;
bbox(4) = bbox(4) + offset2;


function mask = get_mask(I,rawbox,bb,x,sbin,t,hg_size)
I2 = I*0;
rawbox = clip_to_image(slight_expand(rawbox),[1 1 size(I,2) size(I, ...
                                                  1)]);
rawbox = rawbox;

I2(rawbox(2):rawbox(4),rawbox(1):rawbox(3),:) = 1;
Ikeep = I2;
oks = find(I2(:));
I2(oks) = double(randn(size(oks))>0);

I2 = resize(I2,bb(8));
Ikeep = imresize(Ikeep,[size(I2,1) size(I2,2)],'nearest');
Icopy = resize(I,bb(8));


f2 = features_raw(I2,sbin);
f2 = padarray(f2,[t.padder+1 t.padder+1 0]);

%maxers=ceil(size(Ikeep)/sbin)*sbin
%Ikeep(maxers(1),maxers(2),3) = 0;
%f3 = imresize(Ikeep,[size(f2,1) size(f2,2)],'nearest');




f2 = f2(bb(9)+(1:hg_size(1))-1+t.padder,...
        bb(10)+(1:hg_size(2))-1+t.padder,:);


%f3 = f3(bbs{1}(9)+(1:hg_size(1))-1+t.padder,...
%        bbs{1}(10)+(1:hg_size(2))-1+t.padder,:);


fcopy = esvm_features(Icopy,sbin);
fcopy = padarray(fcopy,[t.padder+1 t.padder+1 0]);

fcopy = fcopy(bb(9)+(1:hg_size(1))-1+t.padder,...
              bb(10)+(1:hg_size(2))-1+t.padder,:);

fmask2 = (sum(f2>0,3)>=4);
fmask = sum(f2.^2,3)>0;


[u,v] = find(fmask);
fmask(min(u):max(u),min(v):max(v)) = 1;

mask = (sum(x.^2,3)>0) & fmask;
