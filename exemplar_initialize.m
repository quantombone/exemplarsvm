function exemplar_initialize(cls, mode)
% Initialize script which writes out initial model files for all
% exemplars of a single category from PASCAL VOC trainval set Script
% is parallelizable (and dalalizable!)  
%
% There are several different initialization modes
% DalalMode:  Warp instances to mean frame from in-class exemplars
% Globalmode: Warp instances to hardcoded [8 8] frame
% new10mode:  Use a canonical 8x8 framing of each exemplar (this
%             allows for negative sharing in the future)
%
% cls: VOC class to process
% mode: mode name 
%
% Tomasz Malisiewicz (tomasz@cmu.edu)
VOCinit;

%The sbin size 
SBIN = 8;

%Load default class and mode if no arguments are given
if ~exist('cls','var')
  [cls,mode] = load_default_class;
end

if strmatch(cls,'all')
  classes = VOCopts.classes;
  myRandomize;
  r = randperm(length(classes));
  classes = classes(r);
  for i = 1:length(classes)
    exemplar_initialize(classes{i},mode);
  end
  return;
end

%If this is enabled, then the features will be scaled to the
%canonical dalal-size
dalalmode = 0;
if strfind(mode,'-dt')
  dalalmode = 1;
  fprintf(1,' --##-- DALAL_MODE enabled --##--');
elseif strfind(mode,'-gt')
  dalalmode = 2;
  fprintf(1,' --##-- GLOBAL_DALAL_MODE enabled --##--');
else
  dalalmode = 0;
end

cache_dir =  ...
    sprintf('%s/models/',VOCopts.localdir);

cache_file = ...
    sprintf('%s/%s-%s.mat',cache_dir,cls,mode);

if fileexists(cache_file)
  fprintf(1,'No need to initialize, because models already stored\n');
  return;
end

%Goal ncells gives us a constraint on how many cells we cut the
%object up to (NOTE: not used in new10 function)
GOAL_NCELLS = 100;

%(only non-dalal mode) If greater than one, creates tiny exemplar
%perturbations as additional positives
NWIGGLES = 1; %(NOTE: not used in new10 function)

fprintf(1,'GOAL_NCELLS=%d sbin=%d\n',GOAL_NCELLS,SBIN);

%Only allow display to be enabled on a machine with X
[v,r] = unix('hostname');
if strfind(r, VOCopts.display_machine)==1
  display = 1;
else
  display = 0;
end

fprintf(1,'Class = %s\n',cls);

if ismember(cls,{'all'})
  classes = VOCopts.classes;
  
  r = randperm(length(classes));
  for i = 1:length(classes)
    exemplar_initialize(classes{r(i)});
  end
  return;
end

DTstring = '';
if dalalmode == 1
  %Find the best window size from taking statistics over all
  %training instances of matching class
  hg_size = get_hg_size(cls);
  DTstring = '-dt';
elseif dalalmode == 2
  hg_size = [8 8];
  DTstring = '-gt';
end

results_directory = ...
    sprintf('%s/%s/',VOCopts.localdir,mode);

fprintf(1,'Writing Exemplars of class %s to directory %s\n',cls,results_directory);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'trainval'),...
                  '%s %d');
ids = ids(gt==1);

%randomize ordering of exemplar images if script is ran on cluster
myRandomize;
rrr = randperm(length(ids));
ids = ids(rrr);

for i = 1:length(ids)
  curid = ids{i};

  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
  Ibase = imread(sprintf(VOCopts.imgpath,curid));
  Ibase = im2double(Ibase);
  
  for objectid = 1:length(recs.objects)
    
    %skip difficult objects, and objects not of target class
    if (recs.objects(objectid).difficult==1) | ...
          ~ismember({recs.objects(objectid).class},{cls})
      continue
    end

    fprintf(1,'.');
    
    filer = sprintf('%s/%s.%d.%s.mat',results_directory,curid,objectid,cls);
    filerlock = [filer '.lock'];
    if fileexists(filer) || (mymkdir_dist(filerlock)==0)
      continue
    end
        
    anno = recs.objects(objectid);
    bbox = recs.objects(objectid).bbox;
    gt_box = bbox;
    I = Ibase;

    %Expand the bbox to have some minimum and maximum aspect ratio
    %constraints (if it it too horizontal, expand vertically, etc)
    clear model;
    
    if (dalalmode == 1) || (dalalmode == 2)
      %Do the dalal-triggs anisotropic warping initialization
      model = initialize_model_dt(I,bbox,SBIN,hg_size);
    else
      %Do default exemplar initialization
      %bbox = expand_bbox(bbox,I);
      %model = initialize_model(I,bbox,GOAL_NCELLS,SBIN);
      %model = populate_wiggles(I, model, NWIGGLES);
      if 1
        hg_size = [8 8];
        
        newK = 1;
        [tmp,model] = new10model(I,bbox,SBIN,hg_size,newK,curid);
        
        DO_FRIENDS = 0;
        
        if DO_FRIENDS == 1
          
          [tmp,model2] = new10model(flip_image(I), ...
                                    flip_box(bbox, size(I)), ...
                                    SBIN, hg_size, newK,curid);
          x2 = model2.x;
          t2 = model2.target_id;
          for j = 1:length(t2)
            t2{j}.flip = 1;
            t2{j}.bb = flip_box(t2{j}.bb,size(I));
          end
          
          model.x = cat(2,model.x,x2);
          model.target_id = cat(1,model.target_id,t2);
          
        end
        
        
        model.target_x = model.x;
        
        %% we start with first one
        model.x = model.x(:,1);
      end
    end    
    
    fprintf(1,'Extracting random %d wiggles\n',NWIGGLES);

    %Negative support vectors
    model.svxs = zeros(prod(model.hg_size),0);
    model.svbbs = [];
          
    clear m
    m.curid = curid;
    m.objectid = objectid;
    m.cls = cls;    
    m.gt_box = gt_box;
    m.anno = anno;
    m.model = model;
    m.sizeI = size(I);
    
    %Print the bounding box overlap between the initial window and
    %the final window
    finalos = getosmatrix_bb(m.gt_box, m.model.coarse_box);
    fprintf(1,'Final OS is %.3f\n', ...
            finalos);

    fprintf(1,'final hg_size is %d %d\n',...
            m.model.hg_size(1), m.model.hg_size(2));

    save(filer,'m');
    if exist(filerlock,'dir')
      rmdir(filerlock);
    end

    if display == 1
      figure(1)
      clf
      subplot(1,3,1)
      imagesc(Ibase)
      plot_bbox(m.model.coarse_box,'',[1 0 0],[0 1 0],0,[1 3],m.model.hg_size)
      plot_bbox(m.gt_box,'',[0 0 1])
      axis image
      axis off
      title(sprintf('Image %s.%d',m.curid,m.objectid))
      
      subplot(1,3,2)
      imagesc(m.model.mask);
      axis image
      axis off
      grid on
      title('Mask')
      
      subplot(1,3,3)      
      imagesc(HOGpicture(repmat(m.model.mask,[1 1 features]).*m.model.w))
      axis image
      axis off
      grid on
      title('HOG features')
      drawnow

      pause(.4)
    end
  end  
end

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
%fprintf(1,' Done to %d %d w/h=%.3f h/w=%.3f\n',w,h,w/h,h/w);

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

function target_id = get_target_id(model,I)
%Get the id of the top detection
mmm{1}.model = model;
mmm{1}.model.hg_size = size(model.w);
localizeparams.thresh = -100.0;
localizeparams.TOPK = 1;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;
localizeparams.FLIP_LR = 0;
[rs,t] = localizemeHOG(I,mmm,localizeparams);
target_id = rs.id_grid{1}{1};

function model = populate_wiggles(I, model, NWIGGLES)
%Get wiggles
xxx = replica_hits(I, model.params.sbin, model.target_id, ...
                   model.hg_size, NWIGGLES);

%% GET self feature + NWIGGLES wiggles "perturbed images"
model.x = xxx;
model.w = reshape(mean(model.x,2), model.hg_size);
model.w = model.w - mean(model.w(:));
model.b = -100;

function modelsize = get_bb_stats(h,w)

xx = -2:.02:2;
filter = exp(-[-100:100].^2/400);
aspects = hist(log(h./w), xx);
aspects = convn(aspects, filter, 'same');
[peak, I] = max(aspects);
aspect = exp(xx(I));

% pick 20 percentile area
areas = sort(h.*w);
%TJM: make sure we index into first element if not enough are
%present to take the 20 percentile area
area = areas(max(1,floor(length(areas) * 0.2)));
area = max(min(area, 5000), 3000);

% pick dimensions
w = sqrt(area/aspect);
h = w*aspect;

sbin = 8;
modelsize = [round(h/sbin) round(w/sbin)];



function [hg_size,ids] = get_hg_size(cls)
%% Load ids of all images in trainval that contain cls

VOCinit;
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'trainval'),...
                  '%s %d');
oks = find(gt==1);
ids = ids(oks);
%myRandomize;

fprintf(1,'Computing mean size for class %s\n',cls);
for i = 1:length(ids)
  fprintf(1,'.');
  curid = ids{i};  
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  objects = recs.objects;
  
  goods = find(ismember({objects.class},{cls}) & ([objects.difficult]==0));
  objects = objects(goods);
  bbs{i} = cat(1,objects.bbox);
end

bbs = cat(1,bbs{:});

W = bbs(:,3)-bbs(:,1)+1;
H = bbs(:,4)-bbs(:,2)+1;

[hg_size] = get_bb_stats(H, W);

function model = initialize_model(I,bbox,GOAL_NCELLS,SBIN)
%Get an initial model by cutting out a segment of a size which
%matches the bbox
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
fprintf(1,'hg_size = [%d %d]\n',model.hg_size(1),model.hg_size(2));
model.w = curfeats - mean(curfeats(:));
model.x = curfeats;
model.b = 0;

[model.target_id] = get_target_id(model,I);
model.coarse_box = model.target_id.bb;
%model.mask = (sum(curfeats.^2,3)~=0);

%figure(2)
%imagesc(model.mask)
%drawnow

function [bbox,model] = new10model(I,bbox,SBIN,hg_size,K,curid)
curid = str2double(curid);

rawbox = bbox;
bbox = squareize_bbox(bbox);

%Compute pyramid and all bounding boxes
clear t
params.lpo = 10;
[t.hog, t.scales] = featpyramid2(I, SBIN, params);  
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
       SBIN/t.scales(level) + 1;
  bb(:,3:4) = bb(:,3:4) - 1;
  
  % figure(1)
  % clf
  % imagesc(I)
  % plot_bbox(bb)
  % axis image
  % axis off
  % drawnow   
  allbb{level} = bb;
  alluv{level} = [rawuuu rawvvv];
  alllvl{level} = goodids(:)*0+level;
end

% ip.bb = [([ip.offset(2) ip.offset(1) ip.offset(2)+size(ws{exid},2) ...
%            ip.offset(1)+size(ws{exid},1)] - 1) * ...
%          sbin/ip.scale + 1] + [0 0 -1 -1];

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
  ip.curid = curid;
  
  bb = zeros(1,12);
  bb(1:4) = allbb(superind,:);
  bb(7) = ip.flip;
  bb(8) = ip.scale;
  bb(9:10) = ip.offset;
  bb(11) = curid;
  bb(12) = 0;
  
  bbs{q} = bb;
  curfeats{q} = curfeat;
end

I2 = I*0;
rawbox = clip_to_image(rawbox,[1 1 size(I,2) size(I,1)]);
I2(rawbox(2):rawbox(4),rawbox(1):rawbox(3),:) = 1;
oks = find(I2(:));
I2(oks) = rand(size(oks));
I2 = resize(I2,bbs{1}(8));

% f1 = features(I2,SBIN);
% f1 = padarray(f1,[1 1 0]);
% I3 = imresize(I2,[size(f1,1) size(f1,2)],'nearest');
% I3 = double(I3>0);
% I3 = repmat(I3(:,:,1),[1 1 31]);
% f2 = padarray(I3,[t.padder t.padder 0]);

f2 = features_raw(I2,SBIN);
f2 = padarray(f2,[t.padder+1 t.padder+1 0]);
f2 = f2(bbs{1}(9)+(0:7)+t.padder,...
        bbs{1}(10)+(0:7)+t.padder,:);


fmask = sum(f2.^2,3)>0;

model.coarse_box = allbb(order(1),:);
model.params.sbin = SBIN;
model.hg_size = [hg_size(1) hg_size(2) features];
model.x = curfeats{1};
model.mask = sum(model.x.^2,3)>0 & fmask;
model.w = model.x*0;
mask3 = repmat(model.mask,[1 1 features]);
mask3 = mask3(:);
model.w(mask3) = curfeats{1}(mask3) - mean(curfeats{1}(mask3));

model.b = 0;

model.target_bb = cat(1,bbs{:});
model.x = cellfun2(@(x)reshape(x,[],1),curfeats);
model.x = cat(2,model.x{:});

% bbs = cellfun2(@(x)x.bb,model.target_id);bbs = cat(1,bbs{:});
% rc = rand(size(bbs,1),3);
% for i = 1:size(bbs,1)
%   plot_bbox(bbs(i,:),'',rc(i,:));
%   hold on;
% end
