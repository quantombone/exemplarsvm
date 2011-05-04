function [Iex,alphamask] = get_seg_icon(models,index)

model = models{index};

STEAL_SEGMENTATION = 1;

VOCinit;
filer = sprintf('%s/%s/SegmentationObject/%s.png',VOCopts.datadir, ...
                VOCopts.dataset,model.curid);

filer_class = sprintf('%s/%s/SegmentationClass/%s.png',VOCopts.datadir, ...
                      VOCopts.dataset,model.curid);

  
cmap=VOClabelcolormap;

if ~fileexists(filer)
  Iex = [];
  alphamask = [];
  return;
end
  
  classes = {'aeroplane','bicycle','bird','boat','bottle','bus', ...
             'car','cat','chair','cow','diningtable','dog','horse', ...
             'motorbike','person','pottedplant','sheep','sofa', ...
             'train','tvmonitor'};
  
  clsid = find(ismember(classes,model.cls));
  
  has_seg = 1;
  res = imread(filer);
  res_class = imread(filer_class);
  
  res_only_target_class = res_class==clsid;
  res_class = reshape(cmap(res_class(:)+1,:),size(res_class,1),size(res_class,2),3);
  
  g = model.gt_box;

  fullI = res_class;
  subI = res_class(g(2):g(4),g(1):g(3),:);

  full_res = double(res).*res_only_target_class;
  
  subres = double(res(g(2):g(4),g(1):g(3),:)).* ...
           res_only_target_class(g(2):g(4),g(1):g(3));
  
  us = 0:255;
  

  counts = hist(subres(:),us);
  counts(1) = 0;
  counts(end) = 0;
  [tmp,bestind] = max(counts);
  bestus = us(bestind);
  subres = (subres==bestus);

  Iex = subI;
  alphamask = subres; 
  fprintf(1,'segmentation!!! exists!\n');
  exemplar_overlay.is_class_segmentation = 1;
    
  Iex = fullI;
  alphamask = double(full_res);

% return;
  
  

% resdir = '/nfs/baikal/tmalisie/buslabeling/';
% resfile = sprintf('%s/%s.%d.mat',resdir,model.curid, ...
%                   model.objectid);

% res = load(resfile);
% alphamask = double(res.res.seg>0);
% alphamask = alphamask*.9;
% %figure(4)
% %imagesc(alphamask)
% %pause

% if isfield(models{index},'FLIP_LR') && models{index}.FLIP_LR==1
%   has23 = sum(res.res.seg(:)==2) && sum(res.res.seg(:)==3);
%   if has23
%     new2 = 3;
%     new3 = 2;
%     cur2=find(res.res.seg==2);
%     cur3=find(res.res.seg==3);
    
%     res.res.seg(cur2) = new2;
%     res.res.seg(cur3) = new3;
%   end
  
%   has13 = sum(res.res.seg(:)==1) && sum(res.res.seg(:)==3);
%   if has13
%     new3 = 2;
%     cur3=find(res.res.seg==3);
%     res.res.seg(cur3) = new3;
%   end

%   has12 = sum(res.res.seg(:)==1) && sum(res.res.seg(:)==2);
%   if has12
%     new2 = 3;
%     cur2=find(res.res.seg==2);
%     res.res.seg(cur2) = new2;
%   end

  
% end
% seg2 = faces2colors(res.res.seg);
% Iex = seg2;

% %Iex = flip_image(Iex);
% %alphamask = flip_image(alphamask);
% %end