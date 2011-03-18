function m = train_regressor(m)
VOCinit;
m.models_name = 'nips11';

fg = get_pascal_bg('trainval',m.cls);
%fg = fg(1:40);

localizeparams.thresh = -1;
localizeparams.TOPK = 10;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;
    

%figure(1)
%clf

for qqq = 1:20
c = 1;
clear results
for i = 1:length(fg)
  if c > length(fg)
    return;
  end
  I = convert_to_I(fg{c});
  [a,curid,other] = fileparts(fg{c});
  c = c + 1;
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
  
  goods = find(ismember({recs.objects.class},m.cls));
  goodbbs = cat(1,recs.objects(goods).bbox);
  osmat = getosmatrix_bb(boxes,goodbbs);
  
  maxos = max(osmat,[],2);
  fprintf(1,'maxos %d = %.3f\n',i,max(maxos));
  results{i}.maxos = maxos;
  results{i}.id = maxos*0+i;
  results{i}.id_grid = [rs.id_grid{1}];
  results{i}.x = [];
  if length(rs.support_grid{1})>0
    results{i}.x = cat(2,rs.support_grid{1}{:});  
  end
end

os = cellfun2(@(x)x.maxos,results)
os = cat(1,os{:});
x = cellfun2(@(x)x.x,results);
x = cat(2,x{:});
X = x;



ids = cellfun2(@(x)x.id_grid,results);
ids = cat(2,ids{:});

imageid = cellfun2(@(x)x.id(:),results);
imageid = cat(1,imageid{:});
res = m.model.w(:)'*X-m.model.b;

[aa,bb] = sort(res,'descend');

figure(34)
clf
for q = 1:36
  if q <= length(bb)
    subplot(6,6,q)
    I = convert_to_I(fg{imageid(bb(q))});
    imagesc(I)
    plot_bbox(ids{bb(q)}.bb)
    axis off
    axis image
  end
end

%X(end+1,:) = 0;

%goods = find(os>.2);
%X = X(:,goods);
%os = os(goods);

if 0
  %playing with liblinear here
  SVMC = 100;
  
  y = double(os>.5);
  y(y==0) = -1;
  model = liblinear_train(y, sparse(X)', sprintf(['-s 0 -B 1 -c' ...
                    ' %f'],SVMC));
  wex = model.w(1:end-1)';
  b = -model.w(end);
  if y(1) == -1
    wex = wex*-1;
    b = b*-1;
  end

  m.model.b = b;
  m.model.w = reshape(wex,m.model.hg_size);
end

%X = [X m.model.x];
%os(end+1:end+size(m.model.x,2))=1;

X(end+1,:) = 1;

% [aa,bb] = sort(os+exp(m.model.w(:)'*X(1:end-1,:)-m.model.b)', ...
%                'descend');
%os = os+exp(m.model.w(:)'*X(1:end-1,:)-m.model.b)';

%[aa,bb] = sort(m.model.w(:)'*X(1:end-1,:),'descend');
%X = X(:,bb(1:length(bb)/10));
%os = os(bb(1:length(bb)/10));
[aa,bb] = sort(m.model.w(:)'*X(1:end-1,:),'descend');
[alpha,beta] = sort(bb);
ranks = (beta);

rankscores = ranks.^-.1;
bads = find(os<.1);
os = os .* rankscores';
%os = os';
os = reshape(os,1,[]);
os(bads) = -1;

%w = (os*inv(X'*X+lambda*eye(size(X,2),size(X,2)))*X')';
%w2 = inv(X*X'+lambda*eye(size(X,1),size(X,1)))*X*os';

%alpha = pinv(X)*w;
%alpha2 = pinv(X)*w2;

K = X'*X;

sigmoid = @(x)1./(1+exp(-x));
lambda = 1;
%a = fminunc(@(a)norm(sigmoid(K*a)-os').^2+lambda*norm(X*a)^2,...
%            zeros(size(X,2),1),...
%            optimset('MaxIter',100));

%This is the real dual problem
a = inv(K*K+lambda*K)*K*os';

%get primal detector
w = X*a;

%w = fminunc(@(w)objective(w,X,os),w,optimset('MaxIter',10));

m.model.b = w(end);
m.model.w = reshape(w(1:end-1),m.model.hg_size);
figure(2)
clf
imagesc(HOGpicture(reshape(m.model.w,m.model.hg_size)))
drawnow

figure(444)
plot(w'*X,os,'r.')
drawnow
% if exist('oldos','var')
%   figure(3)
%   plot(oldos,os,'r.')
%   drawnow
% end
%oldos = os;

save m.mat m
end

function [fval,gval] = objective(w,X,os)
lambda = 100;
keyboard
fval = norm(sigmoid(w'*X)-os).^2+lambda*norm(w);
gval = 2*(sigmoid(w'*X)-os)'*sigmoidp(w'*X)*X + 2*lambda*w';

function y = sigmoid(x)
y = 1./(1+exp(-x));

function y = sigmoidp(x)
y=exp(x)./(1+exp(x)).^2;