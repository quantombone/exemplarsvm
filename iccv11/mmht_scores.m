function M = mmht_scores(boxes, maxos, models, neighbor_thresh, count_thresh)
%Given a bunch of detections, learn the M boosting matrix, which
%makes the final scores multiplexed

if ~exist('neighbor_thresh','var')
  neighbor_thresh = 0.5;
end

if ~exist('count_thresh','var')
  count_thresh = 0.5;
end

%already nms-ed within exemplars
if 0
for i = 1:length(boxes)
  boxes{i}(:,5) = 1:size(boxes{i},1);
  boxes{i} = nms_within_exemplars(boxes{i},.5);
  maxos{i} = maxos{i}(boxes{i}(:,5));
  boxes{i}(:,5) = i;
end
end

K = length(models);
N = sum(cellfun(@(x)size(x,2),maxos));

y = cat(1,maxos{:});
os = cat(1,maxos{:})';

scores = cellfun2(@(x)x(:,end)',boxes);
scores = [scores{:}];

xraw = cell(length(boxes),1);
allboxes = cat(1,boxes{:});

% for i = 1:length(boxes)
%   fprintf(1,',');
%   xraw{i} = zeros(length(models),size(boxes{i},1));
% end

%xrawmat = get_box_features_matrix(boxes{i}, K, neighbor_thresh,models);

for i = 1:length(boxes)
  fprintf(1,'.');
  xraw{i} = get_box_features(boxes{i}, K, neighbor_thresh);
end
x = [xraw{:}];

exids = allboxes(:,6);
exids(allboxes(:,7)==1)= exids(allboxes(:,7)==1) + length(models);
imids = allboxes(:,5);

%osmats = cellfun2(@(x)getosmatrix_bb(x,x),boxes);
%thetadiffmats = cellfun2(@(x)getaspectmatrix_bb(x,x),boxes);


fprintf(1,'learning M by counting\n');
tic
%This one works best so far
M = learn_M_counting(x, exids, os, count_thresh);
toc


%lambda = .001;
%w = inv(x*x'+lambda*eye(size(x,1),size(x,1)))*x*os';
%M = learn_M_counting(x, exids, maxos, count_thresh,osmats, ...
%                     thetadiffmats,boxes);
%M = learn_M_gaussian(x, exids, os, count_thresh);

%M = learn_M_counting_learn(x, exids, os, count_thresh);
M.neighbor_thresh = neighbor_thresh;
M.count_thresh = count_thresh;
%d = distSqr_fast(x(:,os>.8),x(:,os>.8));
%gamma =  1/mean(d(:));
%M = learn_M_nonlin_svm(x, allboxes, os, length(models),gamma);
%M = learn_M_ordering(x, allboxes, os, models);

%M = learn_M_perw(x, exids, os);
%M = learn_M_probcounting(x, exids, os);
%M = optimize_M(M,x,exids,os);

r = cell(length(xraw),1);
fprintf(1,'applying boost matrix\n');
tic
for j = 1:length(xraw)
  r{j} = apply_boost_M(xraw{j},boxes{j},M);
end

r = [r{:}];
[aa,bb] = sort(r,'descend');
goods = os>.5;
%lenvec = 1:N;

res = (cumsum(goods(bb))./(1:length(bb)));
M.score = mean(res);
toc

figure(4)
subplot(1,2,1)
plot(scores,os,'r.')
xlabel('singleton scores')
ylabel('OS with gt')

subplot(1,2,2)
plot(r,os,'r.')
xlabel('combined score')
ylabel('os')

figure(5)
clf
[aa,bb] = sort(scores,'descend');
plot(cumsum(os(bb)>.5)./(1:length(os)),'r-')
hold on;
[aa,bb] = sort(r,'descend');
plot(cumsum(os(bb)>.5)./(1:length(os)),'b-')
title('Precision-Recall');
legend('singleton','combined')


function M = learn_M_counting_learn(x, exids, os, count_thresh)
%Learn the matrix by counting activations on positives
N = size(x,2);
K = size(x,1);

C = ones(K,K);

for i = 1:N
  cur = find(x(:,i)>0);  
  C(cur,exids(i)) = C(cur,exids(i)) + os(i)*(os(i) >= count_thresh) / ...
      length(cur);
end

for i = 1:K
  M.w{i} = C(:,i);
  M.b{i} = 0;
end

for xxx = 1:K
fprintf(1,'xxx=%d\n',xxx);
curx = x(:,exids==xxx);
curos = os(exids==xxx);

if max(curos)<.5
  fprintf(1,'skippy\n');
  M.w{xxx} = x(:,1)*0;
  M.b{xxx} = 1;
  continue;
end

extrax = zeros(size(x,1),1);
extrax(xxx) = 1;
extraos = 1;

curx = [curx extrax];
curos = [curos extraos];

cury = double(curos>.5);
cury(cury==0) = -1;

posweights = sum(cury==-1) / sum(cury==1);
model = liblinear_train(cury', sparse(curx)', sprintf(['-s 3 -B 1 -c' ...
                    ' %f -w1 %f'],.01,posweights));

if model.Label(1) == -1
  model.w(1:end-1) = -model.w(1:end-1);
end

model.w(1:end-1) = max(0.0,model.w(1:end-1));

M.w{xxx} = model.w(1:end-1)';
M.b{xxx} = model.w(end);

all_scores = M.w{xxx}'*curx(:,1:end-1) - M.b{xxx};
all_os = curos(1:end-1);

%beta = learn_sigmoid(all_scores, all_os);
%M.w{xxx} = M.w{xxx}*beta(1);
%M.b{xxx} = M.b{xxx}+betboarda(2);
end

function M = learn_M_counting(x, exids, os, count_thresh)
% function M = learn_M_counting(x, exids, maxos, count_thresh, osmats, ...
%                               thetadiffmats, boxes)
%Learn the matrix by counting activations on positives
N = size(x,2);
K = size(x,1);

C = zeros(K,K);
% for i = 1:length(osmats)
%   boxes2 = boxes{i};
%   ids = 1:size(boxes2,1);
%   %boxes2(:,5) = 1:size(boxes2,1);
%   %boxes2 = nms_within_exemplars(boxes2,.5);
%   %ids = boxes2(:,5);
  
%   curosmat = osmats{i}(ids,ids)>.5;
%   curtdm = thetadiffmats{i}(ids,ids);
%   goods = (curosmat > .5);% & curtdm <.1);
 
  
%   goods2 = repmat(maxos{i}>.5,length(maxos{i}),1) & repmat(maxos{i}>.5, ...
%                                                   length(maxos{i}),1)';
  
%   goods = goods .* goods2;

%   %goods = goods - diag(diag(goods));
%   [u,v] = find(goods);
%   eu = boxes2(u,6);
%   ev = boxes2(v,6);
  
%   for j = 1:length(u)
%     C(ev(j),eu(j)) = C(ev(j),eu(j)) + maxos{i}(v(j))*(maxos{i}(v(j))>.5);
    
%     %curosmat(u(i),v(i))*...
%     %    (curosmat(u(i),v(i))>.5);%.*(1- ...
%                                  %  curtdm(u(i),v(i)));
%   end
  
% end

%Cd = diag(C);
%C = C - diag(diag(C));
%C(find(speye(size(C)))) = max(C(:));

for i = 1:N
  cur = find(x(:,i)>0);  
  C(cur,exids(i)) = C(cur,exids(i)) + os(i)*(os(i) >= count_thresh) / ...
      length(cur)*sum(x(:,i));
end

for i = 1:K
  M.w{i} = C(:,i);
  M.b{i} = 0;
end

M.C = sparse(C);


function M = learn_M_gaussian(x, exids, os, count_thresh)
%Learn the matrix by counting activations on positives
N = size(x,2);
K = size(x,1);

C1 = ones(K,K);
C2 = ones(K,K);
for i = 1:N
  cur = find(x(:,i)>0);  
  A = length(cur)*length(cur);
  C1(cur,cur) = C1(cur,cur) + os(i)/A;
  C2(cur,cur) = C2(cur,cur) + 1/A;
end

M.C1 = C1./(C2);

M.C1 = M.C1 / max(M.C1(:));
%maxval = max(M.C1(:));
M.C1(find(eye(size(M,1))))=1.0;



%p1 = zeros(1,size(x,2));
%p2 = zeros(1,size(x,2));
%for i = 1:size(x,2)
%  p1(i) = (x(:,i))' * C1 * (x(:,i));
%  p2(i) = (x(:,i))' * C2 * (x(:,i));  
%end
%p1 = exp(-p1);
%p2 = exp(-p2);


% goods = find(os>=.2);
% bads = find(os>=0);
% mu1 = mean(x(:,goods),2);
% C1 = pinv(cov(x(:,goods)'));

% mu2 = mean(x(:,bads),2);
% C2 = pinv(cov(x(:,bads)'));

% p1 = zeros(1,size(x,2));
% p2 = zeros(1,size(x,2));
% for i = 1:size(x,2)
%   p1(i) = (x(:,i)-mu1)' * C1 * (x(:,i)-mu1);
%   p2(i) = (x(:,i)-mu2)' * C2 * (x(:,i)-mu2);  
% end
% p1 = exp(-p1);
% p2 = exp(-p2);




function M2 = learn_M_counting_golden(x, exids, os)
%Learn the matrix by counting activations on positives
N = size(x,2);
K = size(x,1);

goods = os>.5;

M = ones(K,K);

for i = 1:N
  cur = find(x(:,i)>0);  
  M(cur,exids(i)) = M(cur,exids(i)) + os(i)*(os(i)>=.5)/ ...
      length(cur);
end

for i = 1:K
  M2{i}.w = M(:,i);
  M2{i}.b = 0;
end


function M2 = learn_M_probcounting(x, exids, os)
N = size(x,2);
K = size(x,1);

goods = os>.5;
r = zeros(N,1);

goods = zeros(K,K);
bads = zeros(K,K);

for i = 1:N
  cur = find(x(:,i)>0);
  if os(i) > 0.5
    goods(cur,cur) = goods(cur,cur)+1;
  else
    bads(cur,cur) = bads(cur,cur)+1;
  end
end

counts = goods+bads;
pgood=(goods./(counts+1));

for i = 1:K
  M2{i}.w = pgood(:,i);
  M2{i}.b = 0;
end

function M2 = learn_M_ordering(x, allboxes, os, models)
%Learn a matrix M based on ordering constraints

NMODELS = length(models);

exids = allboxes(:,6);
imids = allboxes(:,5);

%D is feature size, K = number of boxes
D = size(x,1);
K = size(x,2);

superx = sparse(D*NMODELS,K);

for i = 1:K
  superx((1:D) + (exids(i)-1)*D,i) = x(:,i);
end
 
supery = double(os>=.3);
supery(os<.3) = -1;

if 1
  
  %Get all images with a detection with OS greater than .5
  upos = unique(imids(os>.5));
  
  diffx = sparse(D*NMODELS,10000);
  counter = 1;
  tic
  for iii = 1:length(upos)
    curids = find(imids==upos(iii));
    curgoods = find(os(curids) > .5);
    curgoods = curids(curgoods);
    
    osmat = getosmatrix_bb(allboxes(curgoods,1:4), ...
                           allboxes(curids,1:4));
    for a = 1:length(curgoods)
      for b = 1:length(curids)
        if curgoods(a) == curids(b)
          continue
        end
        if osmat(a,b) >= .5 && os(curgoods(a)) > os(curids(b)) + .1
          left = zeros(D*NMODELS,1);
          right = zeros(D*NMODELS,1);
          left((1:D) + (exids(curgoods(a))-1)*D) = x(:,curgoods(a));
          right((1:D) + (exids(curids(b))-1)*D) = x(:,curids(b));
          
          diffx(:,counter) = left-right;
          counter = counter + 1;
        end
      end
    end
  end
  diffx = diffx(:,1:(counter-1));
  diffy = ones(1,size(diffx,2));
  toc
  
  fprintf(1,'GOT %d diffx constraints\n',size(diffx,2));
  newx = [superx diffx];
  newy = [supery diffy];
end

newx = superx(:,supery~=0);
newy = supery(supery~=0);



posweights = sum(newy==-1) / sum(newy==1);

if 1
  tic
  
  model = liblinear_train(newy', (newx)', sprintf(['-s 3 -B 1 -c' ...
                    ' %f -w1 %f'],.1,posweights));
  w = model.w(1:end-1);
  bias = model.w(end);
  
  
  if supery(1) == -1
    w = w*-1;
  end
  toc
end

for i = 1:NMODELS
  M2.w{i} = full(w((1:D) + (i-1)*D)');
  M2.b{i} = bias;
end

function M = optimize_M(M,x,exids,os)

N = size(x,2);
r = zeros(N,1);

randrow = ceil(rand*length(M));
multer = linspace(.1,10,20);
Msave = M;
for q = 1:length(multer)

  M{randrow}.w = Msave{randrow}.w*multer(q);

  for i = 1:N
    r(i) = M{exids(i)}.w'*x(:,i) - M{exids(i)}.b;
  end
  
  [aa,bb] = sort(r,'descend');
  goods = os>.5;
  lenvec = 1:N;
  res = cumsum(goods(bb)>.5)./lenvec;
  score = mean(res);
  values(q) = score;
end

function M2 = learn_M_nonlin_svm(x, allboxes, os, NMODELS, gamma)
%Learn a matrix M based on a nonlinear SVM

exids = allboxes(:,6);
imids = allboxes(:,5);

%D is feature size, K = number of boxes
D = size(x,1);
K = size(x,2);

superx = x;
 
supery = double(os>.5);
supery(supery==0) = -1;

for i=1:NMODELS
  
  newy = supery(exids==i);
  newx = superx(:,exids==i);
  [aa,bb] = sort(newy,'descend');
  newy = newy(bb);
  newx = newx(:,bb);
  newos = os(bb);
  
  if newy(1) == -1
    fprintf(1,'No positives, skipping\n');
    svm_model{i} = [];
    continue
  end
  fprintf(1,'#pos = %d\n',sum(newy==1));
  %posweights = sum(newy==-1) / (sum(newy==1)+eps);
  posweights = 1.0;
  svm_model{i} = libsvmtrain(newy', newx',sprintf(['-s 0 -t 2 -c' ...
                    ' %f -w1 %.9f -g %f -q'],10, posweights,gamma));
  
  %svm_model{i} = svmtrain(newos', newx',sprintf(['-s 3 -t 2 -c' ...
  %                  ' %f -w1 %.9f -p .3 -g %f -q'],.01, posweights,gamma));


  %retvals = svmpredict(newy',newx',svm_model{i});
  %myscores = mysvmpredict(newx,svm_model{i});  
  
end

for i = 1:NMODELS
  %M2{i}.w = full(w((1:D) + (i-1)*D)');
  %M2{i}.b = bias;  
  M2{i}.svm_model = svm_model{i};
end
