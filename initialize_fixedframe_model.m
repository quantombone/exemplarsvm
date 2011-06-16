function model = initialize_fixedframe_model(I, bbox, ...
                                             init_params)
%Initialize exemplars
sbin = init_params.sbin;
hg_size = init_params.hg_size;

K = 1;
ADD_LR = 0;
%Note: here we need to be able to convert curid into a double (only
%works for VOC???)

  % ADD_LR = 0;
  
  % if ADD_LR == 1
    
  %   [tmp,model2] = new10model(flip_image(I), ...
  %                             flip_box(bbox, size(I)), ...
  %                             sbin, hg_size, newK,curid);
  %   x2 = model2.x;
  %   t2 = model2.target_id;
  %   for j = 1:length(t2)
  %     t2{j}.flip = 1;
  %     t2{j}.bb = flip_box(t2{j}.bb,size(I));
  %   end
    
  %   model.x = cat(2,model.x,x2);
  %   model.target_id = cat(1,model.target_id,t2);
    
  % end

rawbox = bbox;

bbox = slight_expand(bbox);
bbox = squareize_bbox(bbox);

%Compute pyramid and all bounding boxes
clear t
params.lpo = 10;
[t.hog, t.scales] = featpyramid2(I, sbin, params);  
t.padder = 2;

allbb = cell(length(t.hog),1);
alluv = cell(length(t.hog),1);
alllvl= cell(length(t.hog),1);
for level = 1:length(t.hog)
  t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], ...
                          0);
  curids = zeros(size(t.hog{level},1),size(t.hog{level},2));
  curids = reshape(1:numel(curids),size(curids));
  goodids = curids(1:size(curids,1)-hg_size(1)+1,1:size(curids,2)- ...
                   hg_size(2)+1);
  [rawuuu,rawvvv] = ind2sub(size(curids),goodids(:));
  uuu = rawuuu - t.padder;
  vvv = rawvvv - t.padder;
  
  bb = ([vvv uuu vvv+hg_size(2) uuu+hg_size(1)] -1) * ...
       sbin/t.scales(level) + 1;
  bb(:,3:4) = bb(:,3:4) - 1;
  
  allbb{level} = bb;
  alluv{level} = [rawuuu rawvvv];
  alllvl{level} = goodids(:)*0+level;
end

alluv = cat(1,alluv{:});
allbb = cat(1,allbb{:});
alllvl = cat(1,alllvl{:});
os = getosmatrix_bb(allbb,bbox);

%% get the symmetry measure
sym1 = abs((allbb(:,1)-bbox(1)) - (allbb(:,3)-bbox(3)));
sym2 = abs((allbb(:,2)-bbox(2)) - (allbb(:,4)-bbox(4)));
meansym = (sym1+sym2)/100;

[tmp,order] = sort(os-meansym,'descend');

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
  %ip.curid = curid;
  
  bb = zeros(1,12);
  bb(1:4) = allbb(superind,:);
  bb(7) = ip.flip;
  bb(8) = ip.scale;
  bb(9:10) = ip.offset;
  %bb(11) = curid;
  bb(12) = 0;
  
  bbs{q} = bb;
  curfeats{q} = curfeat;
end

I2 = I*0;
rawbox = clip_to_image(slight_expand(rawbox),[1 1 size(I,2) size(I,1)]);

I2(rawbox(2):rawbox(4),rawbox(1):rawbox(3),:) = 1;
oks = find(I2(:));
I2(oks) = rand(size(oks));
I2 = resize(I2,bbs{1}(8));

f2 = features_raw(I2,sbin);
f2 = padarray(f2,[t.padder+1 t.padder+1 0]);
f2 = f2(bbs{1}(9)+(1:hg_size(1))-1+t.padder,...
        bbs{1}(10)+(1:hg_size(2))-1+t.padder,:);

fmask = sum(f2.^2,3)>0;

%model.params.sbin = sbin;
model.init_params = init_params;
model.hg_size = [hg_size(1) hg_size(2) features];
%model.coarse_box = allbb(order(1),:);

x = curfeats{1};
model.mask = sum(x.^2,3)>0 & fmask;
model.w = zeros(size(x));

mask3 = repmat(model.mask,[1 1 features]);
mask3 = mask3(:);

%%initialize model to normalized HOG
model.w(mask3) = curfeats{1}(mask3) - mean(curfeats{1}(mask3));
model.b = 0;

model.x = cellfun2(@(x)reshape(x,[],1), curfeats);
model.x = cat(2,model.x{:});

model.bb = cat(1,bbs{:});

%model.target_x = model.x;
%model.x = model.x(:,1);
%model.bb = model.target_bb(1,:);

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

