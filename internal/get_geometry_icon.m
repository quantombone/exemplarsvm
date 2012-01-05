function [Iex,alphamask,faces] = get_geometry_icon(models,index,flip)
%Given a cell array of models and the appropriate index, extract
%the qualitative geometry for the object
%NOTE: only defined for buses

if ~exist('flip','var')
  flip = 0;
end

model = models{index};
if ~strcmp(models{index}.cls,'bus')
  Iex = [];
  alphamask = [];
  faces = [];
  return;
end

resdir = '/nfs/baikal/tmalisie/finalbuslabeling/';
resfile = sprintf('%s/%s.%d.mat',resdir,model.curid, ...
                  model.objectid);

res = load(resfile);
if sum(res.res.seg~=0)==0
  fprintf(1,'warning on this exemplar, needs new anno\n');
  keyboard
end
alphamask = double(res.res.seg>0);
%alphamask = alphamask*.9;
%figure(4)
%imagesc(alphamask)
%pause

if isfield(models{index},'detect_add_flip') && models{index}.detect_add_flip==1
  has23 = sum(res.res.seg(:)==2) && sum(res.res.seg(:)==3);
  if has23
    new2 = 3;
    new3 = 2;
    cur2=find(res.res.seg==2);
    cur3=find(res.res.seg==3);
    
    res.res.seg(cur2) = new2;
    res.res.seg(cur3) = new3;
  end
  
  has13 = sum(res.res.seg(:)==1) && sum(res.res.seg(:)==3);
  if has13
    new3 = 2;
    cur3=find(res.res.seg==3);
    res.res.seg(cur3) = new3;
  end

  has12 = sum(res.res.seg(:)==1) && sum(res.res.seg(:)==2);
  if has12
    new2 = 3;
    cur2=find(res.res.seg==2);
    res.res.seg(cur2) = new2;
  end

  
end
seg2 = faces2colors(res.res.seg);
Iex = seg2;

faces = res.res.faces;


%return;
%Iex = flip_image(Iex);
%alphamask = flip_image(alphamask);
%end

cb = models{index}.gt_box;
PADDER = 0;
%cb = models{mid}.model.coarse_box;

Iex = pad_image(Iex, PADDER);
cb = round(cb + PADDER);
Iex = Iex(cb(2):cb(4),cb(1):cb(3),:);

alphamask = pad_image(alphamask, PADDER);
cb = round(cb + PADDER);
alphamask = alphamask(cb(2):cb(4),cb(1):cb(3),:);

faces = cellfun2(@(y)bsxfun(@minus,y,[cb(1) ...
                    cb(2)]),faces);
  
%allow some transparency
alphamask = alphamask*.8;

if flip == 1
  Iex = flip_image(Iex);
  alphamask = flip_image(alphamask);
  
  faces = ...
      cellfun2(@(x)flip_faces(x,size(Iex)),faces);

end

