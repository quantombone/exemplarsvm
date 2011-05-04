function create_dalal_file;
%Initialize script which writes out initial model files storing the
%positive examples (dalal-triggs style)
% Here we use anisotropic to a single canonical window...
%
%Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;
results_directory = ...
    sprintf('%s/dalals/',VOCopts.localdir);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

classes = VOCopts.classes;
% classes = {'bird','cat','cow','dog','horse','sheep','aeroplane','bicycle',...
%            'boat','bus','car','motorbike','train',...
%            'bottle','chair','diningtable','pottedplant','sofa', ...
%            'tvmonitor','person'};             

%classes = {'bottle'};
classes = {'train'};
%Store exemplars for this class
%if ~exist('cls','var')
%  cls = 'train';
%end

myRandomize;
rrr = randperm(numel(classes));
classes = classes(rrr);

for i = 1:length(classes)

  filer = sprintf('%s/dalal.%s.mat',results_directory,classes{i});
  filerlock = [filer '.lock'];
  if fileexists(filer) | (mymkdir_dist(filerlock)==0)
    continue
  end
  
  m = do_dalal(classes{i},results_directory);
  save(filer,'m');
  rmdir(filerlock);
end

function m = do_dalal(cls, results_directory)
VOCinit;
fprintf(1,'Writing dalals of class %s to directory %s\n',cls,results_directory);

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'trainval'),...
                  '%s %d');
oks = find(gt==1);
ids = ids(oks);
myRandomize;

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

%for iii= 1:size(W)
%  startbb = [1 1 hg_size(2)-1 hg_size(1)-1];
%  endbb = bbs(iii,:);
%  bestos(iii) = get_bestos(startbb,endbb);
%end

fprintf(1,'Class %s found hog size of %d x %d\n',cls,hg_size(1),hg_size(2));

%clear the model
clear m model
model.params.sbin = 8;
model.params.MAX_CELL_DIM = 12;
model.params.MIN_CELL_DIM = 3;
model.params.SVMC = .01;    

model.x = zeros(prod(hg_size)*features,0);
model.hg_size = [hg_size features];

pixels = hg_size * model.params.sbin;
minsize = prod(pixels);
    
fprintf(1,'minsize is %d pixel area\n',minsize);
numskipped = 0;

for i = 1:length(ids)
  curid = ids{i};
  Ibase = imread(sprintf(VOCopts.imgpath,curid));
  Ibase = im2double(Ibase);
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  
  for objectid = 1:length(recs.objects)
    
    %skip difficult objects
    if (recs.objects(objectid).difficult==1) 
      continue
    end
    
    if ~ismember({recs.objects(objectid).class},{cls})
      continue
    end
    
    fprintf(1,'.');
        
    bbox = recs.objects(objectid).bbox;
    I = Ibase;

    %% extend bbox to have the padding
    W = bbox(3)-bbox(1)+1;
    H = bbox(4)-bbox(2)+1;
    
    A = W*H;
    
    startbb = [1 1 hg_size(2)-1 hg_size(1)-1];
    bestos = get_bestos(startbb, bbox);
    
    if A < minsize || bestos<.5
      numskipped = numskipped + 1;
      continue
    end
    
    % extendW = W/hg_size(2);
    % extendH = H/hg_size(1);
    
    % ebox = bbox;
    
    % ebox(1) = ebox(1) - extendW;
    % ebox(3) = ebox(3) + extendW;
    % ebox(2) = ebox(2) - extendH;
    % ebox(4) = ebox(4) + extendH;
    
    % I = pad_image(I,500);
    % ebox = round(ebox+500);
    
    % cropI = I(ebox(2):ebox(4),ebox(1):ebox(3),:);
       
    % hg_extend_size = hg_size + 2;

    % s = size(cropI);
    % s = s(1:2);
    % newscaler = mean(round(s./hg_extend_size));
    % newscaler = max(4,floor(newscaler/4)*4);
    % warpI = max(0.0,min(1.0,imresize(cropI,newscaler*hg_extend_size)));
    % f = features(warpI, newscaler);
    
    %%%% here
    warped = mywarppos(hg_size, Ibase, model.params.sbin, bbox);
    f = features(warped, model.params.sbin);
    
    model.x(:,end+1) = f(:);
    
    %figure(1)
    %clf
    %imagesc(warpI)
    %axis image
    %title(sprintf('sbin = %d, A = %d',newscaler,A));
    %drawnow
    %pause
  end  
end

fprintf(1,'num skipped is %d\n',numskipped);

model.w = reshape(mean(model.x,2),model.hg_size);
model.w = model.w - mean(model.w(:));
model.b = -100;
model.nsv = zeros(prod(model.hg_size),0);
model.svids = [];

model.coarse_box = [];
model.target_id = [];

%m.curid = curid;
%m.objectid = objectid;
m.cls = cls;

%m.gt_box = bbox;
m.model = model;
m.models_name = 'dalal';

%Set up the negative set for this exemplars
%CVPR2011 paper used all train images excluding category images
%m.bg = sprintf('get_pascal_bg(''trainval'',''%s'')',m.cls);
%m.bg = sprintf('get_pascal_bg(''train'',''-%s'')',m.cls);
%m.fg = sprintf('get_pascal_bg(''test'')');


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


function warped = mywarppos(hg_size, I, sbin, bbox)

% warped = warppos(name, model, c, pos)
% Warp positive examples to fit model dimensions.
% Used for training root filters from positive bounding boxes.

pixels = hg_size * sbin;
h = bbox(4) - bbox(2) + 1;
w = bbox(3) - bbox(1) + 1;

cropsize = (hg_size+2) * sbin;

padx = sbin * w / pixels(2);
pady = sbin * h / pixels(1);
x1 = round(bbox(1)-padx);
x2 = round(bbox(3)+padx);
y1 = round(bbox(2)-pady);
y2 = round(bbox(4)+pady);
window = subarray(I, y1, y2, x1, x2, 1);
warped = imresize(window, cropsize, 'bilinear');
