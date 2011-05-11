function [bboxes,scoremasks] = extract_bbs_from_rs(rs, models)
%% Given a resultstruct 'rs' from a localizemeHOG function and a set
%% of models, extract the bounding boxes from the detections

%% Tomasz Malisiewicz (tomasz@cmu.edu)

bboxes = zeros(0,8);
for j = 1:length(rs.id_grid)
  if length(rs.id_grid{j})==0
    continue
  end
  bbs = cellfun2(@(x)x.bb,rs.id_grid{j});
  scores = rs.score_grid{j};
  bbs = cat(1,bbs{:});
  bbs(:,6) = j;
  bbs(:,7) = cellfun(@(x)x.flip,[rs.id_grid{j}])';
  bbs(:,8) = scores';
 
  % for q = 1:length(rs.id_grid{j})
  %   if isfield(rs.id_grid{j}{q},'FLIP_LR') & ...
  %         rs.id_grid{j}{q}.FLIP_LR == 1
  %     bbs(q,6) = bbs(q,6)*-1;
  %   end
  % end
  
  bboxes = [bboxes; bbs];
end
scoremasks = [];
return;

fprintf(1,['WARNING: function not needed as bbs are inside results' ...
           ' now\n']);
error('cannot continue\n');
return;
%Given a sliding window result struct "rs", the scanned window I,
%and the exemplar models, find the scoremasks
VOCinit;

%given the detection results inside the feature pyramid, creating
%bounding boxes

scoremasks = [];

%%% 'bboxes' is [1:4 for bbox, 5 for dummy index, 6 for exemplar
%ids, 7 for scores]
scores = [rs.score_grid{:}];
ids = [rs.id_grid{:}];
if length(ids) == 0
  bboxes = zeros(0,7);
  return;
end

ids = [ids{:}];
exes = [rs.exgrid{:}];
exes = [exes{:}];

bboxes = zeros(length(scores),7);
[aa,bb] = sort(scores,'descend');
scores = scores(bb);
ids = ids(bb);
exes = exes(bb);

ordering = bb;

for i = 1:length(exes)
  scoremasks{i}.scoremask = zeros(1,1);
end

if length(rs.support_grid) > 0 & exist('models','var') & iscell(models)

  sg = cat(2,rs.support_grid{:});
  sg = sg(bb);
  
  for i = 1:length(exes)
    r = models{exes(i)}.model.w(:) .* sg{i};
    rshow = reshape(r,models{exes(i)}.model.hg_size);
    r = sum(rshow,3);% - (sum(models{exes(i)}.model.w .* ...
                     %               reshape(models{exes(i)}.model.x(:,13),size(models{exes(i)}.model.w)),3));
                     %r = -r.^2;
    %r = max(rshow,[],3);
    scoremasks{i}.scoremask=r; 
    
    %scoremasks{i}.scoremask = HOGpicture(reshape(r,models{exes(i)}.model.hg_size));
  end
end

for i = 1:length(scores)
  
  lvl = ids(i).level;
  index = ids(i).index;
  
  [u,v] = ind2sub(rs.rmsizes{lvl}{exes(i)}(1:2),index);

  ratios = sbin/rs.scales(lvl);
  
  if size(models,1) >= 1 & size(models,2) >= 1 & ~iscell(models)
    mask = models;
  else
    mask = ones(models{exes(i)}.model.hg_size);
  end
  u = u - rs.padder;
  v = v - rs.padder;
   
  %NOTE: this has a bug
  %BB = ratios*([(v-1) (u-1) (v+size(mask,2)) ...
  %              (u+size(mask,1))]);
 

  %NOTE: fixed bug .. this seems to be better
  BB = ratios*([(v) (u) (v+size(mask,2)) ...
                (u+size(mask,1))]-1);
  BB(1:2) = BB(1:2) + 1;

  bboxes(i,1:4) = BB;
  bboxes(i,5) = i;
  bboxes(i,6) = exes(i);
  bboxes(i,7) = scores(i); 
end

   
% function BB = adjust_BB(BB,mask)
% BBold = BB;
% starter = [BB(2) BB(1)];
% H = BB(4)-BB(2);
% W = BB(3)-BB(1);
% [subu,subv]=find(mask);

% sm = size(mask);
% minu = min(subu)-1;
% minv = min(subv)-1;
% maxu = max(subu);
% maxv = max(subv);

% starter2 = starter + [H W].*[minu/sm(1) minv/sm(2)];
% ender2 = starter + [H W].*[maxu/sm(1) maxv/sm(2)];
% BB = [starter2(2) starter2(1) ender2(2) ender2(1)];

% function [x1, y1, x2, y2] = rootbox(x, y, scale, padx, pady, rsize)
% x1 = (x-padx)*scale+1;
% y1 = (y-pady)*scale+1;
% x2 = x1 + rsize(2)*scale - 1;
% y2 = y1 + rsize(1)*scale - 1;
