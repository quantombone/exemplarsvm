function m = train_regressor(m)
%Train a linear function which regress well onto other objects and
%has max-margin separation between negatives
VOCinit;
m.models_name = 'nips11';

fg = get_pascal_bg('train',m.cls);

[a,b,c] = fileparts(fg{end});

starter = [a '/'  m.curid c];
fg = setdiff(fg,starter);
fg = [starter; fg];

%(bg) is already inside of m

%[os,xcat,X,ids,imageid] = get_dets(m,fg);  
load myfiles.mat  
os2 = os;
xcat2 = xcat;
X2 = X;
ids2 = ids;
imageid2 = imageid;

  

for qqq = 1:50
  m = learn_model(m,os,xcat,X,ids);
  
  [os,xcat,X,ids,imageid] = collect_top_dets(m,os2,xcat2,X2,ids2, ...
                                           imageid2);

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
    
  res = m.model.w(:)'*X-m.model.b;
  
  [aa,bb] = sort(res,'descend');
  
  figure(34)
  clf
  N = 4;
  for q = 1:N*N
    if q <= length(bb)
      subplot(N,N,q)
      I = convert_to_I(fg{imageid(bb(q))});
      %imagesc(I)
      curbb = ids{bb(q)}.bb;
      
      Ipad = pad_image(I,100);
      curbb = round(curbb+100);
      curI = Ipad(curbb(2):curbb(4),curbb(1):curbb(3),:);
      %imagesc(I)

      if ids{bb(q)}.flip == 1
        curI = flip_image(curI);
      end
      imagesc(curI)
      %plot_bbox(curbb)
      title(sprintf('s=%.3f os=%.3f',aa(q),os(bb(q))))
      axis off
      axis image
    end
  end
  
  % figure(444)
  % plot(res,os,'r.')
  drawnow


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
VOCinit;
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
    
    %goods = find(ismember({recs.objects.class},m.cls));
    gt_classes = {recs.objects.class};
    goods = 1:length(gt_classes);
    goodbbs = cat(1,recs.objects(goods).bbox);
    %goodbbs(end+1,:) = [1 1 1 1];
    %gt_classes(end+1) = '';
    
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

function [os,xcat,X,ids,imageid] = collect_top_dets(m,os,xcat,X,ids, ...
                                                  imageid)
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

function m = learn_model(m,os,xcat,X,ids)
tau = 0;
gamma = .3;
SVMC = .01;

y = double(os>tau & ismember(xcat,m.cls));
y(y==0) = -1;
g = os(y>0);
g = 1./(1+exp(-10*(g-.5)));

%oldscores = m.model.w(:)'*X(:,y>0)-m.model.b;
%g = g.* 1./(1+exp(-1*oldscores))';

%[tmp,index] = max(results{1}.maxos);
%size(X)
%size(m.model.nsv)

X = cat(2,X,m.model.nsv);
y = cat(1,y,-1*ones(size(m.model.nsv,2),1));

index = 1;



[w,b,alphas,pos_inds] = learn_local_capacity(X,y,index,SVMC, ...
                                             gamma,g,m);
%res=w'*X-b;
%plot(res,os,'r.')

m.model.w = reshape(w,m.model.hg_size);
m.model.b = b;
m.model.alphas = alphas;


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
