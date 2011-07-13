function [xrawmat,nbrids,indicator] = get_box_features_matrix(boxes, N, neighbor_thresh,models)
%Get the box features for this particular set of boxes from a
%single image (boxes are the boxes, N is the # of exemplars) and
%the neighbor list (which are the box ids contributing to this
%context feature)

sizes = cellfun(@(x)size(x,1),boxes);
xrawmat = zeros(length(
%The features aggregate information from other exemplar firings,
%such as their overlap score and their scores

% f(b) = [s1 s2 ... sN] where we aggregate features from
% neighboring boxes

%Tomasz Malisiewicz (tomasz@cmu.edu)

% old attempts:
% f(b) = [o1*s1 o2*s2 ... oN*sN]
% f(b) = [o1 o2 ... oN s1 s2 ... sN]

%N is the number of exemplars
%K is the number of boxes
K = size(boxes,1);
x = sparse(N, K);
nbrids = cell(1,K);

%Get overlaps between all boxes in the set
osmat = getosmatrix_bb(boxes, boxes);
 
exid = boxes(:,6)';
uc = unique(exid);

%by adding one to the scores, we effectively get a positive score
scorerow = boxes(:,end)+1;

%scores already calibrated
%scorerow = boxes(:,end);

for j = 1:K
  neighbors = (osmat(:,j) >= neighbor_thresh);
  friend_scores = scorerow.*neighbors;
  friend_os = osmat(:,j).*neighbors;

  nbrids{j} = zeros(length(uc),1);
  counter = 1;
  for q = 1:length(uc)
    oks = find(exid==uc(q));
    [aa,bb] = max(friend_scores(oks));
    
    if aa > 0
      nbrids{j}(counter) = oks(bb);
      counter = counter + 1;
      x(uc(q),j) = aa;
    end
    %curos = friend_os(bb);
    
    %curos = exp(-20*(curos-1).^2);
    %x(uc(q),j) = friend_scores(oks(bb))*friend_os(oks(bb));
    
  end  
  nbrids{j} = nbrids{j}(1:(counter-1));

end
