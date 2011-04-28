function top = adjust_boxes_friends(boxes, models)
%Here we take the detection boxes and adjust them so they are 'GT'
%boxes.

%This step is necessary since the exemplars are framed in a slightly
%different window (one of the coarse aspect ratios based on the 8
%pixel cells) than the actual GT window (which can have any possible
%aspect ratio up to the resolution of the original image)

%% Each detection is an alignment between the "coarse_box" and a
%detection window "d", once we find the rigid transformation
%between "coarse_box" and "d", we can apply the same projection to
%the ground truth window

top = boxes;
%fprintf(1,'.');
for i = 1:size(boxes,1)
  d = boxes(i,:);
  exid = boxes(i,6);
  allc = cellfun2(@(x)x.boxes,models{exid}.friend_info);
  allc = cat(1,allc{:});
  allg = cellfun2(@(x)x.allbbs(x.os_id,:),models{exid}.friend_info);
  allg = cat(1,allg{:});
  
  ws = (exp(allc(:,end)));
  ws = ws / sum(ws);
  allp = zeros(size(allg));
  for j = 1:size(allg,1)
    xform = find_xform(allc(j,1:4),d);
    allp(j,1:4) = apply_xform(allg(j,1:4),xform);
  end
  
  bb3 = size(allp,1)*mean(allp.*repmat(ws,1,4),1);
  top(i,1:4) = bb3;
  
  %os = getosmatrix_bb(top(i,:),d);
  %fprintf(1,'os=%.3f\n',os);
  %n = norm(apply_xform(c,xform) - d(1:4));
  %if (n>.001)
  %  keyboard
  %end
end
