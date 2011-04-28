function recs2 = prune_recs(recs,class1,class2)
%Keep only annotations which show both objects overlapping

if ~exist('class2','var')
  class1 = 'person';
  class2 = 'motorbike';
end
recs2 = recs;
for i = 1:length(recs)
  c = {recs(i).objects.class};
  hits1 = ismember(c,{class1});
  if sum(hits1) == 0
    recs2(i).objects = recs2(i).objects(1:-1);
    continue;
  end
  
  hits2 = ismember(c,{class2});
  if sum(hits2) == 0
    recs2(i).objects = recs2(i).objects(1:-1);
    continue;
  end
  
  bbs = cat(1,recs(i).objects.bbox);
  hits1 = find(hits1);
  hits2 = find(hits2);
  bb1 = bbs(hits1,:);
  bb2 = bbs(hits2,:);
  
  osmat = getosmatrix_bb(bb1,bb2);
  
  %keyboard
  
  vals = osmat(:);
  hits = find(vals>.05);
  
  %[maxval,ind] = max(osmat(:));

  if length(hits) == 0
    recs2(i).objects = recs2(i).objects([]);
    continue;
  end
  [u,v] = ind2sub(size(osmat),hits);
  
  %keep u,v here
  keepers = [hits1(u') hits2(v')];
  %if maxval > .05
    recs2(i).objects = recs2(i).objects(keepers);
  %else
  %  recs2(i).objects = recs2(i).objects([]);
  %end
end