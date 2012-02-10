function Isv = esvm_show_det_stack(bbs, train_set, K2, K1, m)
% Create a [K1]x[K2] image which visualizes the detection windows, as
% well as information about the trained exemplar [m].
% The first row shows [exemplar image, w+, w- ,
%    mean0, mean 1, ... ,mean N]
% Second row first icon starts the top detections. This
% visualization is used to show top negative support vectors as
% well as top detection from any set.
% Inputs:
%    bbs: a set of bounding boxes, where bbs(j,11) is the image
%    from the j-th bounding box. images are shown in the raw order
%    train_set: the set of (virtual) images which bbs(:,11) refers to
%    [K2,K1]: sizes of grid
%    m: the model, if present fills in icon and w+/w- pics
%
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm
% NOTE(TJM): there is a bug when K2=15, K1=20, with the averages
% not being only in the first row of the image

if ~exist('K2','var')
  K1 = 5;
  K2 = 5;
end

if ~exist('K1','var')
  K1 = K2;
end

K1 = max(K1,5);
K2 = max(K2,5);

%% sort by score
% if isfield(m.model,'svxs') && (numel(m.model.svxs)>0)
%   if isfield(m.mining_params,'dfun') && (m.mining_params.dfun == 1)
%     r = m.model.w(:)'*bsxfun(@minus,m.model.svxs,m.model.x(:,1)).^2 ...
%         - m.model.b;
%   else
%     r = m.model.w(:)'*m.model.svxs - m.model.b;
%   end
%   [aa,bb] = sort(r,'descend');
%   bbs = bbs(bb, :);
%   m.model.svxs = m.model.svxs(:, bb);
% else

% end


if ~exist('m','var')
  pos_picture = ones(10,10,1);
  neg_picture = ones(10,10,1);
else
  pos_picture = (HOGpicture(m.w));
  neg_picture = (HOGpicture(-m.w));
end

N = size(bbs,1);
N = min(N,K1*K2);
bbs = bbs(1:N,:);

if N > 0
  ucurids = unique(bbs(:,11));
else
  ucurids = [];
end

ims = cell(N,1);

for i = 1:length(ucurids)

  Ibase = toI(train_set{ucurids(i)});
  
  hits = find(bbs(:,11)==ucurids(i));
  for j = 1:length(hits)
    
    cb = bbs(hits(j),:);
    
    d1 = max(0,1 - cb(1));
    d2 = max(0,1 - cb(2));
    d3 = max(0,cb(3) - size(Ibase,2));
    d4 = max(0,cb(4) - size(Ibase,1));
    mypad = max([d1,d2,d3,d4]);
    PADDER = round(mypad)+2;
    I = pad_image(Ibase,PADDER);
    
    bb = round(cb + PADDER);
    ims{hits(j)} = I(bb(2):bb(4),bb(1):bb(3),:);
   
    if cb(7) == 1
      ims{hits(j)} = flip_image(ims{hits(j)});
    end
        
  end
end

%Get the exemplar frame icon
if exist('m','var')
  [~,~,Ibase] = esvm_get_exemplar_icon({m},1);
else
  Ibase = ones(10,10,3);
end

newsize = [size(Ibase,1) size(Ibase,2)];
newsize = 100/newsize(1) * newsize;
newsize = round(newsize);
newsize = newsize + 10;

ims = cellfun2(@(x)max(0.0,min(1.0,imresize(x,newsize))),ims);



NSHIFT_BASE = 3;

imstack = cat(4,ims{:});


SSS = size(imstack,4)*~isempty(imstack);

NMS = K2-NSHIFT_BASE;
if NMS == 1
  cuts(1) = SSS;
else
  cuts = round(linspace(1,SSS,NMS+1));
  cuts = cuts(2:end);
end

PADSIZE = 5;

for i = 1:length(cuts)
  %NOTE(TJM): mss is turned off because of times 0
  if SSS > 0
    mss{i} = mean(imstack(:,:,:,1:cuts(i)),4)*0+1;
  else
    mss{i} = zeros(newsize(1),...
                   newsize(2),3)*0+1;
  end
end


%KKK = K1-3;
ims = cellfun2(@(x)pad_image(x,PADSIZE,[1 1 1]),ims);

if length(ims)<K1*K2
  ims{K1*K2} = zeros(newsize(1)+PADSIZE*2,...
                       newsize(2)+PADSIZE*2,3);
end

for j = (N+1):(K1*K2)
  ims{j} = zeros(newsize(1)+PADSIZE*2,...
                   newsize(2)+PADSIZE*2,3);
end


pos_picture = jettify(imresize(pos_picture,newsize,'nearest'));
neg_picture = jettify(imresize(neg_picture,newsize,'nearest'));


%first four slots are reserved for the image
NSHIFT = length(mss) + NSHIFT_BASE;
ims((NSHIFT+1):end) = ims(1:end-NSHIFT);

%exemplar icon goes in slot 1
ims{1} = max(0.0,...
               min(1.0,imresize(Ibase,...
                                [size(pos_picture,1),...
                    size(pos_picture, ...
                         2)])));
ims{2} = pos_picture;
ims{3} = neg_picture;

ims(NSHIFT_BASE+(1:length(mss))) = mss;

for q = 1:(NSHIFT_BASE+length(mss))
  ims{q} = pad_image(ims{q},PADSIZE,[1 1 1]);
end

ims = reshape(ims,K1,K2)';
for j = 1:K2
  svrows{j} = cat(2,ims{j,:});
end

Isv = cat(1,svrows{:});
