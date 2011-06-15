function model = initialize_model_dt(I,bbox,SBIN,hg_size)
%Get an initial model by cutting out a segment of a size which
%matches the bbox

warped = mywarppos(hg_size, I, SBIN, bbox);
curfeats = features(warped, SBIN);
model.x = curfeats(:);    
model.params.sbin = SBIN;

model.hg_size = size(curfeats);    
model.w = curfeats - mean(curfeats(:));
model.b = 0;
model.coarse_box = bbox;

function [hg_size,ids] = get_hg_size(cls, sbin)
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

[hg_size] = get_bb_stats(H, W, sbin);


function modelsize = get_bb_stats(h,w, sbin)

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

%sbin = 8;
modelsize = [round(h/sbin) round(w/sbin)];
