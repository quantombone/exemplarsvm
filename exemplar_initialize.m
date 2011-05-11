function exemplar_initialize(cls,mode)
%% Initialize script which writes out initial model files for all
%% exemplars of a single category from PASCAL VOC trainval set
%% Script is parallelizable (and dalalizable!)
%% Tomasz Malisiewicz (tomasz@cmu.edu)
VOCinit;

%The sbin size 
SBIN = 8;

%Load default class and mode if no arguments are given
if ~exist('cls','var')
  [cls,mode] = load_default_class;
end

cache_dir =  ...
    sprintf('%s/models/',VOCopts.localdir);

cache_file = ...
    sprintf('%s/%s-%s.mat',cache_dir,cls,mode);

if fileexists(cache_file)
  fprintf(1,'No need to initialize, because models already stored\n');
  return;
end

%If this is enabled, then the features will be scaled to the
%canonical dalal-size
if strfind(mode,'-dt')
  dalalmode = 1;
else
  dalalmode = 0;
end

if dalalmode == 1
  fprintf(1,' --##-- DALAL_MODE enabled --##--');
end

%Goal ncells gives us a constraint on how many cells we cut the
%object up to
GOAL_NCELLS = 100;

%(only non-dalal mode) If greater than one, creates tiny exemplar
%perturbations as additional positives
NWIGGLES = 1;

fprintf(1,'GOAL_NCELLS=%d sbin=%d\n',GOAL_NCELLS,SBIN);

%Only allow display to be enabled on a machine with X
[v,r] = unix('hostname');
if strfind(r, VOCopts.display_machine)==1
  display = 1;
else
  display = 0;
end

display = 0;


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
  %hg_size = [7 10];
end

results_directory = ...
    sprintf('%s/%s/',VOCopts.localdir,mode);

fprintf(1,'Writing Exemplars of class %s to directory %s\n',cls,results_directory);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

% results_directory = [results_directory cls '/']
% if ~exist(results_directory,'dir')
%   fprintf(1,'Making directory %s\n',results_directory);
%   mkdir(results_directory);
% end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'trainval'),...
                  '%s %d');
ids = ids(gt==1);

%randomize ordering of exemplar images if script is ran on cluster
myRandomize;
rrr = randperm(length(ids));
ids = ids(rrr);

%figure(1)
%clf
%imagesc(Ibase)
%plot_bbox(recs.objects(objectid).bbox)
%title(sprintf('i=%d objectid=%d',i,objectid));
%pause
%continue

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
        
    bbox = recs.objects(objectid).bbox;
    gt_box = bbox;
    I = Ibase;

    %Expand the bbox to have some minimum and maximum aspect ratio
    %constraints (if it it too horizontal, expand vertically, etc)
    clear model;
    
    if dalalmode == 1
      %Do the dalal-triggs anisotropic warping initialization
      model = initialize_model_dt(I,bbox,SBIN,hg_size);
    else
      bbox = expand_bbox(bbox,I);
      %Do default exemplar initialization
      model = initialize_model(I,bbox,GOAL_NCELLS,SBIN);
      model = populate_wiggles(I, model, NWIGGLES);
    end
    
    fprintf(1,'Extracting random %d wiggles\n',NWIGGLES);

    
    %Negative support vectors
    model.nsv = zeros(prod(model.hg_size),0);
    model.svids = [];
    
    %Validation support vectors
    model.vsv = zeros(prod(model.hg_size),0);
    model.vsvids = [];
    
    %Friend support vectors
    model.fsv = zeros(prod(model.hg_size),0);
    model.fsvids = [];
      
    clear m
    m.curid = curid;
    m.objectid = objectid;
    m.cls = cls;    
    m.gt_box = gt_box;
    m.model = model;
    m.sizeI = size(I);
    
    %Print the bounding box overlap between the initial window and
    %the final window
    finalos = getosmatrix_bb(m.gt_box, m.model.coarse_box);
    fprintf(1,'Final OS is %.3f\n', ...
            finalos);


    save(filer,'m');
    if exist(filerlock,'dir')
      rmdir(filerlock);
    end

    if display == 1
      figure(1)
      clf
      imagesc(Ibase)
      plot_bbox(m.model.coarse_box,'',[1 0 0])
      plot_bbox(m.gt_box,'',[0 0 1])
      axis image
      title(sprintf('%s.%d',m.curid,m.objectid))
      drawnow
      pause(.5)
    end
  end  
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
  masker{a} = max(0.0,min(1.0,imresize(I2,[size(f_real{a},1) size(f_real{a}, ...
                                                  2)])));
  [tmpval,ind] = max(masker{a}(:));
  masker{a} = (masker{a}>.1);
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
[f_real,scales] = featpyramid2(I_real_pad,model.params.sbin, 10);

%Extract the region from each level in the pyramid
[masker,sizer] = get_matching_masks(f_real, I2);

%Now choose the mask which is closest to N cells
[targetlvl, mask] = get_ncell_mask(GOAL_NCELLS, masker, ...
                                                sizer);
[uu,vv] = find(mask);
curfeats = f_real{targetlvl}(min(uu):max(uu),min(vv):max(vv),:);
model.hg_size = size(curfeats);    

model.w = curfeats - mean(curfeats(:));
model.b = 0;

[model.target_id] = get_target_id(model,I);
model.coarse_box = model.target_id.bb;
