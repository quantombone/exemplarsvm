function [fboxes,cs] = transfer_friends(models, bboxes)
%transfer friends onto detections

VOCinit;
% if isfield(models{i},'FLIP_LR') && models{i}.FLIP_LR == 1
%   models{i}.gt_box = flip_box(models{i}.gt_box,[recs.size.height ...
%                     recs.size.width]);
% end

hasfriends = cellfun(@(x)size(x.friendbb,1),models);
hasfriends = find(hasfriends);

fboxes = cell(1,length(bboxes));
cs = cell(1,length(bboxes));

for i = 1:length(bboxes)
  curb = bboxes{i};
  goods = ismember(curb(:,6),hasfriends);
  curb = curb(goods,:);
  
  fboxes{i} = zeros(0,7);
  cs{i} = cell(0,1);
  for j = 1:size(curb,1)
    exid = curb(j,6);
    curg = models{exid}.gt_box;
    others = models{exid}.friendbb;
    
    if isfield(models{exid},'FLIP_LR') && models{exid}.FLIP_LR == 1
      sizeI = models{exid}.sizeI;      
      %curg = flip_box(curg,sizeI);
      others = flip_box(others,sizeI);
    end

    xform_cd = find_xform(curg, curb(j,:));
    
    %news = others;
    
    if isfield(models{exid},'FLIP_LR') && models{exid}.FLIP_LR == 1
      others = flip_box(others,sizeI);
    end

    news = apply_xform(others,xform_cd);
    
    %if isfield(models{exid},'FLIP_LR') && models{exid}.FLIP_LR == 1
    %  news = flip_box(news,sizeI);
    %end

       
    news(:,5) = j;
    news(:,6) = curb(j,6);
    news(:,7) = curb(j,end);
    fboxes{i} = [fboxes{i}; news];

    cs{i} = [cs{i} models{exid}.friendclass];

  end
  
end
