function model = initialize_goalsize_model(I,bbox,init_params)
GOAL_NCELLS = init_params.GOAL_NCELLS;
SBIN = init_params.SBIN;
%Get an initial model by cutting out a segment of a size which
%matches the bbox

%Expand the bbox to have some minimum and maximum aspect ratio
%constraints (if it it too horizontal, expand vertically, etc)
bbox = expand_bbox(bbox,I);

I2 = zeros(size(I,1),size(I,2));    
I2(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;
model.params.sbin = SBIN;

%% NOTE: why was I padding this at some point and now I'm not???
ARTPAD = 0; %120;
I_real_pad = pad_image(I,ARTPAD);

%Get the hog features (+wiggles) from the ground-truth bounding box
params.lpo = 10;
[f_real,scales] = featpyramid2(I_real_pad, model.params.sbin, params);

%Extract the region from each level in the pyramid
[masker,sizer] = get_matching_masks(f_real, I2);

%Now choose the mask which is closest to N cells
[targetlvl, mask] = get_ncell_mask(GOAL_NCELLS, masker, ...
                                                sizer);
[uu,vv] = find(mask);
curfeats = f_real{targetlvl}(min(uu):max(uu),min(vv):max(vv),:);

model.hg_size = size(curfeats);
model.mask = logical(ones(model.hg_size(1),model.hg_size(2)));

fprintf(1,'hg_size = [%d %d]\n',model.hg_size(1),model.hg_size(2));
model.w = curfeats - mean(curfeats(:));
model.b = 0;

model.x = curfeats;
model.bb = get_target_bb(model,I);

function [targetlvl,mask] = get_ncell_mask(GOAL_NCELLS, masker, ...
                                                        sizer)
%Get a the mask and features, where mask is closest to NCELL cells
%as possible

MAXDIM = 10;
fprintf(1,'maxdim is %d\n',MAXDIM);
for i = 1:size(masker)
  [uu,vv] = find(masker{i});
  if ((max(uu)-min(uu)+1) <= MAXDIM) && ...
        ((max(vv)-min(vv)+1) <= MAXDIM)
    targetlvl = i;
    mask = masker{targetlvl};
    return;
  end
end
fprintf(1,'didnt find a match\n');
%Default to older strategy
ncells = prod(sizer,2);
[aa,targetlvl] = min(abs(ncells-GOAL_NCELLS));
mask = masker{targetlvl};

function [masker,sizer] = get_matching_masks(f_real, I2)
%Given a feature pyramid, and a segmentation mask inside I2, find
%the best matching region per level in the feature pyramid

masker = cell(length(f_real),1);
sizer = zeros(length(f_real),2);

for a = 1:length(f_real)
  goods = double(sum(f_real{a}.^2,3)>0);
  
  masker{a} = max(0.0,min(1.0,imresize(I2,[size(f_real{a},1) size(f_real{a}, ...
                                                  2)])));
  [tmpval,ind] = max(masker{a}(:));
  masker{a} = (masker{a}>.1) & goods;

  if sum(masker{a}(:))==0
    [aa,bb] = ind2sub(size(masker{a}),ind);
    masker{a}(aa,bb) = 1;
  end
  [uu,vv] = find(masker{a});
  masker{a}(min(uu):max(uu),min(vv):max(vv))=1;
  sizer(a,:) = [range(uu)+1 range(vv)+1];
end

function bbox = expand_bbox(bbox,I)
%Expand region such that is still within image and tries to satisfy
%these constraints best
%requirements: each dimension is at least 50 pixels, and max aspect
%ratio os (.25,4)
for expandloop = 1:10000
  % Get initial dimensions
  w = bbox(3)-bbox(1)+1;
  h = bbox(4)-bbox(2)+1;
  
  if h > w*4 || w < 50
    %% make wider
    bbox(3) = bbox(3) + 1;
    bbox(1) = bbox(1) - 1;
  elseif w > h*4 || h < 50
    %make taller
    bbox(4) = bbox(4) + 1;
    bbox(2) = bbox(2) - 1;
  else
    break;
  end
  
  bbox([1 3]) = cap_range(bbox([1 3]), 1, size(I,2));
  bbox([2 4]) = cap_range(bbox([2 4]), 1, size(I,1));      
end


function target_bb = get_target_bb(model,I)
%Get the id of the top detection
mmm{1}.model = model;
mmm{1}.model.hg_size = size(model.w);
localizeparams.thresh = -100000.0;
localizeparams.TOPK = 1;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;
localizeparams.FLIP_LR = 0;
localizeparams.pyramid_padder = 5;
[rs,t] = localizemeHOG(I,mmm,localizeparams);
target_bb = rs.bbs{1}(1,:);
