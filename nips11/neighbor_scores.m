function [maxscores,maxlevels,maxoffsets,maxos,maxfeats] = ...
    neighbor_scores(g, w, b, gtmask)
%Get neighbor scores across all instances

order = 1:length(g);

maxscores = zeros(length(order),1);
maxlevels = zeros(length(order),1);
maxoffsets = zeros(length(order),2);
maxos = zeros(length(order),1);
maxfeats = zeros(prod(size(w)),length(order));

for iii = 1:length(order)
  i = order(iii);
  
  bestoffset = zeros(length(g{i}.curw),2);
  scores = zeros(length(g{i}.curw),1);
  osscores = zeros(length(g{i}.curw),1);
  curfeats = zeros(prod(size(w)),length(g{i}.curw));
  
  for j = 1:length(g{i}.curw)
    pads = [ceil(size(w,1)/2) ceil(size(w,2)/2) 0];
    curx = padarray(g{i}.curw{j},pads,0);
    curm = padarray(g{i}.curm{j},pads,0);
    rm = fconv(curx,{w},1,1);
    [value,index] = max(rm{1}(:));
    
    scores(j) = max(rm{1}(:));

    
    [uu,vv] = ind2sub(size(rm{1}),index);
    [offset] = [uu vv] - pads(1:2) - 1;
    bestoffset(j,:) = offset;
    
    resm = zeros(size(curm));
    resm(uu:uu+size(w,1)-1,vv:vv+size(w,2)-1,:) = gtmask;
    osscores(j) = resm(:)'*curm(:);
    osscores(j) = osscores(j)/(sum(resm(:))+sum(curm(:))- ...
                               osscores(j));

    xxx = curx(uu:uu+size(w,1)-1,vv:vv+size(w,2)-1,:);
    curfeats(:,j) = xxx(:); 
  end
  

  [maxscores(iii),ind] = max(scores);
  maxlevels(iii) = ind;
  maxos(iii) = osscores(ind);
  maxoffsets(iii,:) = bestoffset(ind,:);
  maxfeats(:,iii) = curfeats(:,ind);
  fprintf(1,'.');
end

maxscores = maxscores - b;
