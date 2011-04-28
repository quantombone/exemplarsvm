function [A] = perform_x2w_ridge_regression
%% Perform Ridge Regression to estimate W (classifier space) from X
%% (HOG space) by learning a per-cell mapping.

%% The mapping is performed by loading some pre-trained exemplars,
%% treating the cells independely and then performing a regression

resfile = '/nfs/baikal/tmalisie/mappingA.mat';
if fileexists(resfile)
  load(resfile);
  return;
end

%% Load some random exemplars
BASEDIR = '/nfs/baikal/tmalisie/voc2/local/VOC2007/exemplars/mined/';
files = dir([BASEDIR '10*mat']);
rrr = randperm(length(files));
files = files(rrr(1:500));

X = [];
W = [];

for i = 1:length(files)
  fprintf(1,'.');
  m = load([BASEDIR files(i).name]);
  w = m.m.model.w;
  x = m.m.model.x(:,13);
  x = reshape(x, size(w));
  x = reshape(x, [], 31)';
  w = reshape(w, [], 31)';
  X = [X x];
  W = [W w];
end

X(end+1,:) = 1;
%A = W*pinv(X);
A = (inv(X*X' + .000001*eye(32))*X*W')';

save(resfile,'A');

return;
%W2 = A*X; 
%[tmp,A] = do_r(models{3}.model.x, models{3}.model.w);

c = 1;
N = length(models);
for i = 1:length(models)
  w = models{i}.model.w;
  x = models{i}.model.x;
  
  x = reshape(x,size(w));
  X = reshape(x,[],31)';
  X(end+1,:) = 1;
  W = reshape(w,[],31)';
  
  W2 = A*X;
  W2 = W2';
  
  W2 = reshape(W2,size(w));
 
  diff = sqrt(sum((w - W2).^2,3));
 
  figure(1)
  clf
  subplot(1,4,c)
  imagesc(HOGpicture(x))
  title(num2str(i))
  subplot(1,4,c+1)
  imagesc(HOGpicture(w))
  subplot(1,4,c+2)
  imagesc(HOGpicture(W2))
  subplot(1,4,c+3)
  imagesc(diff)
  pause
  %c = c+3;
end

function [W2,A] = do_r(x, w)

%D = 3;
%N = 100;
%X = randn(D,N);
%A = randn(D,D);
%W = A*X;

x = reshape(x,size(w));
X = reshape(x,[],31)';
W = reshape(w,[],31)';

X(end+1,:) = 1;

A = W*pinv(X);
W2 = A*X;
W2 = W2';

W2 = reshape(W2,size(w));
%plot(w(:),W2(:),'r.')

return;
figure(1)
subplot(1,3,1)
imagesc(HOGpicture(x))
subplot(1,3,2)
imagesc(HOGpicture(w))
subplot(1,3,3)
imagesc(HOGpicture(W2))