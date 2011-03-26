function m = train_regressor(m)
%Train a linear function which regress well onto other objects and
%has max-margin separation between negatives
VOCinit;
m.models_name = 'nips11';


savem = m;
fg_train = get_pascal_bg('train',m.cls);
fg_val = get_pascal_bg('val',m.cls);

%remove self
[a,b,c] = fileparts(fg_train{end});
starter = [a '/'  m.curid c];
fg_train = setdiff(fg_train,starter);
fg_val = setdiff(fg_val,starter);

fg = [starter; fg_train];
fgraw = fg;

fg2 = get_pascal_bg('train',['-' m.cls]);

VOCopts.localdir = [VOCopts.localdir '/myfiles'];

finalI=sprintf('%s/Zfinal_%s.%05d.png',VOCopts.localdir,m.curid, ...
               m.objectid);
if fileexists(finalI)
  return;
end

%(bg) is already inside of m
for qqq = 1:1

  resfile = sprintf('%s/result.%s.%05d.mat',VOCopts.localdir,...
                  m.curid,m.objectid);
  
  filerlock = [resfile '.lock'];
    %fprintf(1,'hack not checking\n');

  if ~fileexists(resfile)% (mymkdir_dist(filerlock) == 0) %|| 
    continue
  end

  
  file1 = sprintf('%s/myfiles_fgraw.%s.%05d.mat',VOCopts.localdir,...
                  m.curid,m.objectid);
  file2 = sprintf('%s/myfiles_fg_val.%s.%05d.mat',VOCopts.localdir,...
                  m.curid,m.objectid);
  try
    load(file1);% myfiles_fgraw.mat  
    load(file2);% myfiles_fg_val.mat
  catch
    [os,xcat,X,ids,imageid] = get_dets(m,fgraw);  
    save(file1,'os','xcat','X','ids','imageid');
    
    [osT,xcatT,XT,idsT,imageidT] = get_dets(m,fg_val);
    save(file2,'osT','xcatT','XT','idsT','imageidT');
    %save(sprintf('myfiles_%05d.mat',qqq),'os','xcat','X','ids','imageid');
  end

for q = 1:length(ids)
  ids{q}.curid = imageid(q);
end

for q = 1:length(idsT)
  idsT{q}.curid = imageidT(q);
end

os2 = os;
xcat2 = xcat;
X2 = X;
ids2 = ids;
imageid2 = imageid;

g2 = compute_gain_vector(m,os2,xcat2,X2,ids2,fgraw);

[osN,xcatN,XN,idsN,imageidN,bg] = get_dets_neg(m);  
gN = compute_gain_vector(m,osN,xcatN,XN,idsN,bg);

os = os2;
xcat = xcat2;
X = X2;
ids = ids2;
imageid = imageid2;
g = g2;
%[os,xcat,X,ids,imageid,g] = collect_top_dets(m,os2,xcat2,X2,ids2, ...
%                                             imageid2,g2);


os = cat(1,os,osN);
xcat = cat(1,xcat,xcatN);
X = cat(2,X,XN);
ids = cat(2,ids,idsN);
imageid = cat(1,imageid,imageidN);
fg = cat(1,fg,fg2);  

g = cat(1,g,gN);

  %g = compute_gain_vector(m,os,xcat,X,ids);
  

  [m] = learn_model(m,os,xcat,X,ids,g);
  %os(end+1:end+size(m.model.x,2))=1;
  
  %X(end+1,:) = 1;
  
  % [aa,bb] = sort(os+exp(m.model.w(:)'*X(1:end-1,:)-m.model.b)', ...
  %                'descend');
  %os = os+exp(m.model.w(:)'*X(1:end-1,:)-m.model.b)';
  
  %[aa,bb] = sort(m.model.w(:)'*X(1:end-1,:),'descend');
  %X = X(:,bb(1:length(bb)/10));
  %os = os(bb(1:length(bb)/10));
  
  % [aa,bb] = sort(m.model.w(:)'*X(1:end-1,:),'descend');
  % [alpha,beta] = sort(bb);
  % ranks = (beta);
  
  % rankscores = ranks.^-.1;
  % bads = find(os<.2);
  % os = os .* rankscores';
  % os = reshape(os,1,[]);
  % os(bads) = 0;
  
  % %w = (os*inv(X'*X+lambda*eye(size(X,2),size(X,2)))*X')';
  % %w2 = inv(X*X'+lambda*eye(size(X,1),size(X,1)))*X*os';
  
  % %alpha = pinv(X)*w;
  % %alpha2 = pinv(X)*w2;
  
  % K = X'*X;
  
  % sigmoid = @(x)1./(1+exp(-x));
  % lambda = 1;
  % %a = fminunc(@(a)norm(sigmoid(K*a)-os').^2+lambda*norm(X*a)^2,...
  % %            zeros(size(X,2),1),...
  % %            optimset('MaxIter',100));
  
  % %This is the real dual problem
  % a = inv(K*K+lambda*K)*K*os';
  
  % %get primal detector
  % w = X*a;
  
  % %w = fminunc(@(w)objective(w,X,os),w,optimset('MaxIter',10));
  
  % m.model.b = w(end);
  % m.model.w = reshape(w(1:end-1),m.model.hg_size);
  
  % figure(2)
  % clf
  % imagesc(HOGpicture(reshape(m.model.w,m.model.hg_size)))
  % drawnow
  

  
  % targetX = X2;
  % targetids = ids2;
  % targetg = g2;
  
  
  targetX = X;
  targetids = ids;
  targetset = 'train';
  targetfg = fg;

  % [os3,xcat3,targetX,targetids,imageid3,targetg] = collect_top_dets(m,os,xcat,X,ids, ...
  %                                                 imageid,g);
  


  %[aa,bb] = sort(targetg,'descend');
  
  
  %figure(2)
  %plot(res,targetg,'r.')
  %xlabel('resulting score')
  %ylabel('g')
  %drawnow
  Isv = cell(0,1);
  %% output of current algorithm  
  targetw = m.model.w(:);
  targetb = m.model.b;
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg,'maxcapacity',targetset);
  
  return;
  %% output of original algorithm (exemplarsvm)
  targetw = savem.model.w(:);
  targetb = savem.model.b;
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg,'exemplarsvm',targetset);
  
  %% output of w-centering trick classifier
  targetw = mean(m.model.x,2);
  targetb = 0;
  targetw = targetw - mean(targetw(:));
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg, ...
                'zeromean',targetset);
  
  targetw = mean(m.model.x,2);
  targetb = 0;
  targetfun = @(x,y)distSqr_fast(x,y);
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg, ...
                'nndist',targetset,targetfun);

  
  targetw = mean(m.model.x,2);
  targetb = 0;
  targetfun = @(x,y)sum((x-y).^2./(x+y+eps));
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg, ...
                'chisq',targetset,targetfun);

  
  
  targetX = XT;
  targetids = idsT;
  targetset = 'val';
  targetfg = fg_val;
 
  % [os3,xcat3,targetX,targetids,imageid3,targetg] = collect_top_dets(m,os,xcat,X,ids, ...
  %                                                 imageid,g);
  


  %[aa,bb] = sort(targetg,'descend');
  
  
  %figure(2)
  %plot(res,targetg,'r.')
  %xlabel('resulting score')
  %ylabel('g')
  %drawnow

  %% output of current algorithm  
  targetw = m.model.w(:);
  targetb = m.model.b;

  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg,'maxcapacity',targetset);
  

  %% output of original algorithm (exemplarsvm)
  targetw = savem.model.w(:);
  targetb = savem.model.b;
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg,'exemplarsvm',targetset);
  
  %% output of w-centering trick classifier
  targetw = mean(m.model.x,2);
  targetb = 0;
  targetw = targetw - mean(targetw(:));
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg, ...
                'zeromean',targetset);
  
  targetw = mean(m.model.x,2);
  targetb = 0;
  targetfun = @(x,y)distSqr_fast(x,y);
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg, ...
                'nndist',targetset,targetfun);

  
  targetw = mean(m.model.x,2);
  targetb = 0;
  targetfun = @(x,y)sum((x-y).^2./(x+y+eps));
  Isv{end+1} = show_and_save(m,targetw,targetb,targetX,targetids,targetfg, ...
                'chisq',targetset,targetfun);

  

  % %% custom for chisq nn-dist
  % mx = mean(m.model.x,2);
  % for qqq = 1:size(targetX,2)
  %   chisq = (targetX(:,qqq)-mx);
  %   chisq = sum((chisq).^2 ./(targetX(:,qqq)+mx));
  %   d(qqq) = chisq;
  % end
  % %d = distSqr_fast(mx,targetX);
  
  % res = -d;
  % [aa,bb] = sort(res,'descend');

  % for i = 1:length(targetids)
  %   targetids{i}.FLIP_LR = 0; 
  % end
  % targetids = cellfun2(@(x)rmfield(x,'FLIP_LR'),targetids);
  % Isv = get_sv_stack([targetids{bb(1:min(length(bb),Nstack^2))}],targetfg);
  % figure(34)
  % clf
  % imagesc(Isv)
  % imwrite(Isv,sprintf('%s/dets_chisq-%s_%s.%05d.png',VOCopts.localdir,targetset,m.curid,m.objectid));

  % %% custom for euclidean nn-dist
  % mx = mean(m.model.x,2);
  % d = distSqr_fast(mx,targetX);
  % res = -d;
  % [aa,bb] = sort(res,'descend');

  % for i = 1:length(targetids)
  %   targetids{i}.FLIP_LR = 0; 
  % end
  % targetids = cellfun2(@(x)rmfield(x,'FLIP_LR'),targetids);
  % Isv = get_sv_stack([targetids{bb(1:min(length(bb),Nstack^2))}],targetfg);
  % figure(34)
  % clf
  % imagesc(Isv)
  % imwrite(Isv,sprintf('%s/dets_nndist-%s_%s.%05d.png',VOCopts.localdir,targetset,m.curid,m.objectid));

  % %% custom for chisq nn-dist
  % mx = mean(m.model.x,2);
  % for qqq = 1:size(targetX,2)
  %   chisq = (targetX(:,qqq)-mx);
  %   chisq = sum((chisq).^2 ./(targetX(:,qqq)+mx));
  %   d(qqq) = chisq;
  % end
  % %d = distSqr_fast(mx,targetX);
  
  % res = -d;
  % [aa,bb] = sort(res,'descend');

  % for i = 1:length(targetids)
  %   targetids{i}.FLIP_LR = 0; 
  % end
  % targetids = cellfun2(@(x)rmfield(x,'FLIP_LR'),targetids);
  % Isv = get_sv_stack([targetids{bb(1:min(length(bb),Nstack^2))}],targetfg);
  % figure(34)
  % clf
  % imagesc(Isv)
  % imwrite(Isv,sprintf('%s/dets_chisq-%s_%s.%05d.png',VOCopts.localdir,targetset,m.curid,m.objectid));


  
  % figure(444)
  % plot(res,os,'r.')

  
  III=cat(1,Isv{:});

  fprintf(1,'writing %s\n',finalI);
  imagesc(III)
drawnow
  imwrite(III,finalI);



  %save(resfile,'m');
  %rmdir(filerlock);
  


  %save m.mat m
end

% function [fval,gval] = objective(w,X,os)
% lambda = 100;
% keyboard
% fval = norm(sigmoid(w'*X)-os).^2+lambda*norm(w);
% gval = 2*(sigmoid(w'*X)-os)'*sigmoidp(w'*X)*X + 2*lambda*w';

% function y = sigmoid(x)
% y = 1./(1+exp(-x));

% function y = sigmoidp(x)
% y=exp(x)./(1+exp(x)).^2;

function [os,xcat,X,ids,imageid] = get_dets(m,fg)
%Given a set of images inside fg and an exemplar inside m, get the
%detections by applying exemplar to within-class images

%os is the returned similarity score metric

VOCinit;

%get self annotation
r = PASreadrecord(sprintf(VOCopts.annopath,m.curid));
bbs = cat(1,r.objects.bbox);
os = getosmatrix_bb(m.gt_box,bbs)
[alpha,ind] = max(os);

fself = get_feature_vector(r,ind);

localizeparams.thresh = -1;
localizeparams.TOPK = 10;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;

c = 1;
for i = 1:length(fg)
  if c > length(fg)
    return;
  end
  
  I = convert_to_I(fg{c});
  
  [a,curid,other] = fileparts(fg{c});
  
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
  [rs] = localizemeHOG(I,{m},localizeparams);
  
  %extract detection box vectors from the localization results
  [coarse_boxes] = extract_bbs_from_rs(rs, {m});
  
  boxes = coarse_boxes;
  %map GT boxes from training images onto test image
  boxes = adjust_boxes(coarse_boxes,{m});
  
  
  % figure(1)
  % % clf
  % subplot(3,3,i)
  % imagesc(I)
  % plot_bbox(boxes)
  % drawnow
  % axis image
  % axis off
  
  %use all objects in the annotation
  gt_classes = {recs.objects.class};
  goods = 1:length(gt_classes);
  goodbbs = cat(1,recs.objects(goods).bbox);
    
  osmat = getosmatrix_bb(boxes,goodbbs);
  
  [maxos,maxind] = max(osmat,[],2);
  maxcat = gt_classes(maxind)';
  
  if length(rs.support_grid{1})>0
    maxval = max(m.model.w(:)'*cat(2,rs.support_grid{1}{:})- ...
                 m.model.b);
  else
    maxval = -1;
  end
  fprintf(1,'maxos %d = %.3f, maxval=%.3f\n',c,max(maxos),maxval);
  results{c}.maxos = maxos;
  results{c}.maxcat = maxcat;
  results{c}.id = maxos*0+c;
  results{c}.id_grid = [rs.id_grid{1}];
  results{c}.x = [];
  if length(rs.support_grid{1})>0
    results{c}.x = cat(2,rs.support_grid{1}{:});  
  end
  
  c = c + 1;
end


os = cellfun2(@(x)x.maxos,results);
os = cat(1,os{:});

xcat = cellfun2(@(x)reshape(x.maxcat,[],1),results);
xcat = cat(1,xcat{:});

x = cellfun2(@(x)x.x,results);
x = cat(2,x{:});
X = x;

ids = cellfun2(@(x)x.id_grid,results);
ids = cat(2,ids{:});

imageid = cellfun2(@(x)x.id(:),results);
imageid = cat(1,imageid{:});
  
function [os,xcat,X,ids,imageid,bg] = get_dets_neg(m)

VOCinit;
bg = get_pascal_bg('train',['-' m.cls]);

c = 1;
for i = 1:length(m.model.svids)
  
  %I = convert_to_I(fg{c});
  cursv = m.model.svids{i};
  [a,curid,tmp]=fileparts(bg{cursv.curid});    
  %[a,curid,other] = fileparts(fg{c});
    
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
  % [rs] = localizemeHOG(I,{m},localizeparams);
  
  % %extract detection box vectors from the localization results
  % [coarse_boxes] = extract_bbs_from_rs(rs, {m});
 
  %bbs = cellfun2(@(x)x.bb,m.model.svids); 
  bbs = m.model.svids{i}.bb;
  %coarse_boxes = cat(1,bbs{:});
  coarse_boxes = bbs;
  %flipids = cellfun(@(x)x.flip,m.model.svids);
  flipids = m.model.svids{i}.flip;

  
  boxes = coarse_boxes;
  %map GT boxes from training images onto test image

  boxes(:,5) = 0;
  boxes(:,6) = 1;
  boxes(:,7) = flipids;
  boxes(:,8) = m.model.w(:)'*m.model.nsv(:,i) - m.model.b;
  coarse_boxes = boxes;
  
  boxes = adjust_boxes(coarse_boxes,{m});

    
    % figure(1)
    % % clf
    % subplot(3,3,i)
    % imagesc(I)
    % plot_bbox(boxes)
    % drawnow
    % axis image
    % axis off
    
    %goods = find(ismember({recs.objects.class},m.cls));
    gt_classes = {recs.objects.class};
    goods = 1:length(gt_classes);
    goodbbs = cat(1,recs.objects(goods).bbox);
    %goodbbs(end+1,:) = [1 1 1 1];
    %gt_classes(end+1) = '';
    
    osmat = getosmatrix_bb(boxes,goodbbs);
    
    [maxos,maxind] = max(osmat,[],2);
    maxcat = gt_classes(maxind)';
    
    maxval = max(m.model.w(:)'*m.model.nsv(:,i)-...
                 m.model.b);
    
    fprintf(1,'maxos %d = %.3f, maxval=%.3f\n',c,max(maxos),maxval);
    results{c}.maxos = maxos;
    results{c}.maxcat = maxcat;
    results{c}.id = maxos*0+c;
    results{c}.id_grid = m.model.svids{i};%[rs.id_grid{1}];
    results{c}.x = m.model.nsv(:,i);
    
    c = c + 1;
  end

  
  os = cellfun2(@(x)x.maxos,results);
  os = cat(1,os{:});
  
  xcat = cellfun2(@(x)reshape(x.maxcat,[],1),results);
  xcat = cat(1,xcat{:});
  
  x = cellfun2(@(x)x.x,results);
  x = cat(2,x{:});
  X = x;
  
  
  ids = cellfun2(@(x)x.id_grid,results);
  ids = cleanse_ids(ids);


  %ids = cat(2,ids{:});
  
  imageid = cellfun2(@(x)x.id(:),results);
  imageid = cat(1,imageid{:});


function [os,xcat,X,ids,imageid,g] = collect_top_dets(m,os,xcat,X,ids, ...
                                                  imageid,g)


res = m.model.w(:)'*X-m.model.b;
%% get top det per image
uq = unique(imageid);
keepers = zeros(size(X,2));

for i = 1:length(uq)
  cur = find(imageid==uq(i));
  [aa,bb] = max(res(cur));
  keepers(cur(bb))=1;
end
inds = find(keepers);
os = os(inds);
X = X(:,inds);
xcat = xcat(inds);
ids = ids(inds);
imageid = imageid(inds);
g = g(inds);

% %% add negatives here
% X = cat(2,X,m.model.nsv);
% %y = cat(1,y,-1*ones(size(m.model.nsv,2),1));
% g = cat(1,g,zeros(size(m.model.nsv,2),1));

% %% must get os, xcat, and imageid from the negatives

% bg = get_pascal_bg('train',['-' m.cls]);
% for i = 1:length(m.model.svids)
%   cursv = m.model.svids{i};
%   [a,curid,tmp]=fileparts(bg{cursv.curid});
%   gts = recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
%   keyboard
% end

function [m] = learn_model(m,os,xcat,X,ids,g)
tau = .3;

SVMC = .01;
gamma = 1.0;
gamma = .023;

%[aa,bb] = sort(g,'descend');
%y = os*0-1;
%y(bb(1:10)) = 1;

y = double(os>tau);% & ismember(xcat,m.cls));

y(y==0) = -1;

%g = 1./(1+exp(-10*(g-.5)));
%oldscores = m.model.w(:)'*X(:,y>0)-m.model.b;
%g = g.* 1./(1+exp(-1*oldscores))';

%[tmp,index] = max(results{1}.maxos);
%size(X)
%size(m.model.nsv)

index = 1;

w = m.model.w(:);
b = m.model.b;
r = [];

%[w,b,alphas,pos_inds] = learn_local_capacity(X,y,index,SVMC, ...
%                                             gamma,g(y==y(index)),m);


[w,b,r,pos_inds] = learn_local_rank_capacity(X,y,index,SVMC, ...
                                             gamma,g,m);

%res=w'*X-b;
%plot(res,os,'r.')

m.model.w = reshape(w,m.model.hg_size);
m.model.b = b;
m.model.r = r;%alphas = alphas;


% if 0
%   %playing with liblinear here
%   SVMC = 100;

%   y = double(os>.5);
%   y(y==0) = -1;
%   model = liblinear_train(y, sparse(X)', sprintf(['-s 0 -B 1 -c' ...
%                   ' %f'],SVMC));
%   wex = model.w(1:end-1)';
%   b = -model.w(end);
%   if y(1) == -1
%     wex = wex*-1;
%     b = b*-1;
%   end

%   m.model.b = b;
%   m.model.w = reshape(wex,m.model.hg_size);
% end

function g = compute_gain_vector(m,os_big,xcat,X,ids,fg)
%Here we compute the gain vector between an exemplar (inside of m)
%and the detection window.  it is a measure of annotation
%similarity which generalizes overlap score and category equality

VOCinit;

%get self annotation
r = PASreadrecord(sprintf(VOCopts.annopath,m.curid));
bbs = cat(1,r.objects.bbox);
os = getosmatrix_bb(m.gt_box,bbs)
[alpha,ind] = max(os);
fself = get_feature_vector(r,ind);

% f = zeros(length(fself),length(os));
% for i = 1:length(os_big)
%   curfile = fg{ids{i}.curid};
%   [tmp,curid,tmp] = fileparts(curfile);
%   r = PASreadrecord(sprintf(VOCopts.annopath,m.curid));
%   bbs = cat(1,r.objects.bbox);
%   os = getosmatrix_bb(ids{i}.bb,bbs);
%   [alpha,ind] = max(os);
%   f(:,i) = get_feature_vector(r,ind);
% end

%d = distSqr_fast(fself,f)';
g = os_big*100;% + 20*double(ismember(xcat,m.cls));
%g = g - d*20;
g(os<.2) = -100;



function targetids = cleanse_ids(targetids)
for i = 1:length(targetids)
  targetids{i}.FLIP_LR = 0; 
end
targetids = cellfun2(@(x)rmfield(x,'FLIP_LR'),targetids);


function Isv = show_and_save(m,targetw,targetb,targetX,targetids,fg, ...
                       titler,targetset,targetfun);

VOCinit;
VOCopts.localdir = [VOCopts.localdir '/myfiles'];

if exist('targetfun','var')
  d = zeros(size(targetX,2),1);
  for i = 1:size(d,1)
    d(i) = targetfun(targetw,targetX(:,i));
  end
  res = -d;
else
  res = targetw'*targetX-targetb;
end

[aa,bb] = sort(res,'descend');

%% bb is now the ordering, but nms it

newids = cellfun(@(x)x.curid,targetids(bb));
uq = unique(newids);
for i = 1:length(uq)
  curhit = find(newids==uq(i));
  saver = bb(curhit(1));
  bb(curhit) = 0;
  bb(curhit(1))=saver;
end

bb = bb(bb>0);

Nstack = 4;
for i = 1:length(targetids)
  targetids{i}.FLIP_LR = 0; 
end
targetids = cellfun2(@(x)rmfield(x,'FLIP_LR'),targetids);
curm = m;
curm.model.w = reshape(targetw,size(curm.model.w));
Isv = get_sv_stack([targetids{bb(1:min(length(bb),12))}],fg,curm);
figure(34)
clf
imagesc(Isv)
axis image
title([titler ' ' targetset])
imwrite(Isv,sprintf('%s/dets_%s-%s_%s.%05d.png',VOCopts.localdir,titler,targetset,m.curid, ...
                    m.objectid));

function f = get_feature_vector(r, ind)

VOCinit;

bbs = cat(1,r.objects.bbox);
maxos = getosmatrix_bb(bbs,bbs(ind,:));
maxos(ind) = 0;

classes = {r.objects.class};

[aa,bb]=ismember(classes(setdiff(1:length(classes),ind)),VOCopts.classes);
relative_counts = hist(bb,1:20);

insiders = find(maxos>.5);
[aa,bb]=ismember(classes(insiders),VOCopts.classes);
friend_counts = hist(bb,1:20);

self_counts = double(ismember(VOCopts.classes,classes(ind)));

sub = r.objects(ind);
views = {'Frontal','Rear','Left','Right','Other'};
vvector = double(ismember(views,sub.view));
vvector = [vvector sub.truncated sub.occluded];
cvec = double(ismember(VOCopts.classes,sub.class));

f=cat(1,vvector(:),cvec(:),friend_counts(:),relative_counts(:));

if 0
W = bbs(ind,3)-bbs(ind,1)+1;
H = bbs(ind,4)-bbs(ind,2)+1;
scale = 100/W;
W = W*scale;
H = H*scale;

hs = linspace(10,5,250);
hs = (exp(-.01*(hs-H).^2));
hs = hs/sum(hs);
end
