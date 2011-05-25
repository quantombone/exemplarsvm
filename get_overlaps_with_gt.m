function [overlaps,matchind,matchclass] = get_overlaps_with_gt(m, objids, bg)
%% Given an exemplar in m, and its detections in objids, and a set of
%% images in bg: find the overlaps with all instances of the category
%% belonging to m (inside m.cls)
%% 
%% Tomasz Malisiewicz (tomasz@cmu.edu)

if length(objids) == 0
  objids = [m.model.svids{:}];
end

if iscell(objids)
  objids = [objids{:}];
end
overlaps = zeros(length(objids),1);
matchind = zeros(length(objids),1);
matchclass = zeros(length(objids),1);


%load each image at once
uids = unique([objids.curid]);
VOCinit;

for i = 1:length(uids)
  [tmp,curid] = fileparts(bg{uids(i)});
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  gt_bb = cat(1,recs.objects.bbox);
  [tmp,curclass] = ismember({recs.objects.class},VOCopts.classes);
  %hits = find(ismember({recs.objects.class},m.cls));
  %if length(hits) == 0
  %  continue
  %end
  %gt_bb = gt_bb(hits,:);
  
  hits = find([objids.curid]==uids(i));
  startbb = cat(1,objids(hits).bb);
  endbb = startbb;
  %if ~isfield(m,'models_name') || strcmp(m.models_name,'dalal') ==
  %0
  
  for j = 1:size(startbb,1)
    xform = find_xform(m.model.coarse_box, startbb(j,:));
    endbb(j,:) = apply_xform(m.gt_box, xform);
  end
  %end
  osmat = getosmatrix_bb(endbb,gt_bb);
  [maxos,maxind] = max(osmat,[],2);
  matchind(hits) = maxind;
  overlaps(hits) = maxos;
  matchclass(hits) = curclass(maxind);
end
