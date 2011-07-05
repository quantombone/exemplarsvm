function [overlaps,matchind,matchclass] = get_overlaps_with_gt(m, objids)
% Given an exemplar in m, and a set of detections objids (defaults
% to m.model.svids), find the overlaps, by propagating the GT box
% within m, and measure overlap with all GT instances in the
% detection's image
% 
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('objids','var') || (length(objids) == 0)
  try
    objids = [m.model.svids{:}];
  catch
    fprintf(1,'Error svids are not same type?\n');
    keyboard
  end
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
  curid = sprintf('%06d',uids(i));

  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  gt_bb = cat(1,recs.objects.bbox);
  [tmp,curclass] = ismember({recs.objects.class},VOCopts.classes);
  
  hits = find([objids.curid]==uids(i));
  startbb = cat(1,objids(hits).bb);
  endbb = startbb;
  
  for j = 1:size(startbb,1)
    xform = find_xform(m.model.coarse_box, startbb(j,:));
    endbb(j,:) = apply_xform(m.gt_box, xform);
  end

  osmat = getosmatrix_bb(endbb,gt_bb);
  [maxos,maxind] = max(osmat,[],2);
  matchind(hits) = maxind;
  overlaps(hits) = maxos;
  matchclass(hits) = curclass(maxind);
end
