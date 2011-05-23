function keepers = nms_objid(objid)
%Give a stack of objids, assume they are already sorted and perform
%NMS across detections from the same image, ...

curids = cellfun(@(x)x.curid, objid);
bbs = cellfun2(@(x)x.bb, objid);
bbs = cat(1,bbs{:});
ucurids = unique(curids);

keepers = curids*0;
for i = 1:length(ucurids)
  hits = find(curids==ucurids(i));
  if length(hits) == 1
    %no need for nms with one detection
    keepers(hits) = 1;
    continue
  end
  
  b = bbs(hits,:);
  b(:,5) = 1:size(b,1);
  b(:,8) = size(b,1):-1:1;
  [res] = nms(b,.5);

  keepers(hits(res(:,5)))=1;

end

keepers = find(keepers);

