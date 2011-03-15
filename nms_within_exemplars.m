function top = nms_within_exemplars(boxes,os_thresh)

if length(boxes) == 0
  top = boxes;
  return;
end
uex = unique(boxes(:,6));
top = cell(length(uex),1);
for i = 1:length(uex)
  top{i} = nms(boxes(boxes(:,6)==uex(i),:),os_thresh);
end
top = cat(1,top{:});

[aa,bb] = sort(top(:,end),'descend');
top = top(bb,:);
