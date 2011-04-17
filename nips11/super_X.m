function super_X(X,y,ids)
%% Here we try to take a bunch of X's (a subset of categories from
%% SUN397 and apply the super objective to it
%fprintf(1,'getting global pca ');
%[v,d,mu] = mypca(X,20);
%fprintf(1,'done\n');

% N = 50;
% bg = get_movie_bg('/nfs/hn22/tmalisie/exemplarsvm/data/faceapt.mov', ...
%                   N);

% X = zeros(1984,N);
% y = zeros(N,1);
% ids = bg;
% for i = 1:N
%   fprintf(1,'.');
%   curI = convert_to_I(bg{i});
%   curI = imresize(curI,[200 200]);
%   f = features(curI,20);
%   X(:,i) = f(:);
%   y(i) = 1;
% end

targety = 1;

%choose phone booths and control towers
targety = [259 255 388 307 1:10];
%targety = [259 255 307 112 32 44 56];
targety = [111 123 34 88 38 223 43 98];
%targety = [123 98];
%targety = 1:10;
%targety = [259 388 43];

[aa,bb] = ismember(y,targety);
goods = find(aa==1);
others = find(y<=0);

ids = ids([goods; others]);
X2 = X(:,others);
X = X(:,goods);
X = cat(2,X,X2);
y = y([goods; others]);

params.learning_rate = .0001;

%capacity gain coefficient
params.gamma = 0;

%how much to separate positives and negatives
params.lambda = 10;

%how much to regularize the norms of w 
params.lambda2 = .001;

%how much to force nearby w to be close on L2 sense
params.sigma = 0; %.01;

%how much to force max-self constraint
params.theta = 10;
params.topfactor = 1;

W = X;
mx = mean(X,2);
for i = 1:size(W,2)
  W(:,i) =X(:,i);
  W(:,i) = W(:,i) - mean(W(:,i));
end
W = W*0;


N = size(X,2);
A = eye(N,N);

%d = distSqr_fast(X,X);
%A = (d<.5*median(d(:)));
%A(find(speye(size(A))))=1;


  cx = repmat(y(:),1,length(y));
  cy = cx';
  Asave = A*0;
  Asave(cx==cy) = 1;
  A = Asave;

shower = zeros(100,100,3);
X(end+1,:) = 1;
W(end+1,:) = 0;

for q = 1:2000


  if mod(q,100) == 0
    fprintf(1,',');
  end
  
  for chunks = 1:10
  
  %obj = evaluate_objective(W,A,params,X);
  %grad = evaluate_gradient(W,A,X,1);
  
  
  r = [randperm(size(W,2))];% randperm(size(W,2));];

  %r = repmat(r,100,1);
  
  %r = r(1:min(length(r),2000));
    
  %r = 1;
  %r = r(1);
  %r = ones(100,1)*2;

  for z = 1:length(r)
    grad = evaluate_gradient(W,A,params,X,r(z));
    W(:,r(z)) = W(:,r(z)) - params.learning_rate*grad;
  end
  

if 0
  %use gamma soft parameter
  hmat = h(W'*X);
  hnmat = h(-W'*X);
  wmat = distSqr_fast(W,W);
  
  A = double(2*params.sigma*wmat + params.lambda*hmat +params.lambda*hmat'- params.lambda*hnmat -params.lambda*hnmat'<2*params.gamma);
  
 % A = double(hmat+hmat'<2*params.gamma);
  A(find(speye(size(A))))=1;
  
  A = (A+A')/2;  
else
  
  A = A*0;
  for q = 1:size(W,2)
    cur = W(:,q)'*X;
    gains = double(y==y(q));
    gains(gains==0) = -1;
    [aa,bb] = sort(cur,'descend');
    
    cs = cumsum(gains(bb));
    [alpha,beta] = max(cs);
    
    %beta = [beta];
    %purity = mean(y(bb(1:beta)) == y(q));
    %fprintf(1,'beta is %d purity is %.3f\n',beta,purity);
    
    A(q,bb(1:beta)) = 1;
    A(q,q) = 1;
    
  end
  A = double(A&A');
  
end
end

%use saved class-based A
A = Asave;

figure(1)
clf

if mod(q,100)==0
  %shower = create_grid_res(ids(bb),4);
end

clf
subplot(2,2,1)
imagesc(W'*X)
title(sprintf('W''X Matrix iter %d',q))

subplot(2,2,2)
r = W'*X;
sigmoid = @(x)(1./(1+exp(-x)));
imagesc(sigmoid(.2*r))

title('matrix after sigmoids')

subplot(2,2,3)

%fprintf(1,'showing memex graph\n');
if 1 %q < 1000
  imagesc(A)% .*( W'*X))
else
  tic
  I = show_memex(A,ids,y);
  toc
  imagesc(I)
end
[uu,vv] = find(A);
goods = find(uu~=vv);
corr = mean(y(uu(goods))==y(vv(goods)));
title(num2str(corr*100))
subplot(2,2,4)
plot(r(2,:));

drawnow


end


function obj = evaluate_objective(W,A,params,X)
%evaluate the objective function, slow but handy

%gain matrix is identity
G = ones(size(A));


N = size(X,2);
obj = 0;
for i = 1:N
  for j = 1:N
    if A(i,j) ~= 0
      obj = obj + params.sigma*norm(W(:,i)-W(:,j)).^2;

      obj = obj + params.lambda*(h(W(:,i)'*X(:,j))-h(-W(:,i)'*X(:,j))) - ...
            params.gamma*G(i,j);
    end
    
    obj = obj + params.lambda*h(-W(:,i)'*X(:,j)) + params.theta*h(params.topfactor*W(:,i)'*(X(:,i)-X(:,j)));
  end
  obj = obj + params.lambda2*norm(W(:,i)).^2;
end

function obj = evaluate_gradient(W,A,params,X,k)
%evaluate the objective function, slow but handy

%gain matrix is identity
G = ones(size(A));

N = size(X,2);
obj = zeros(size(X,1),1);

r = randperm(N);
for j = r(1:3) %ceil(rand*N) %1:N
  if A(k,j) ~= 0
    obj = obj + params.sigma*2*(W(:,k)-W(:,j));
    
    obj = obj + params.lambda*(hprime(W(:,k)'*X(:,j))*X(:,j)-hprime(-W(:,k)'*X(:,j))*-X(:,j));
  end
  
  obj = obj + params.lambda*hprime(-W(:,k)'*X(:,j))*-X(:,j) + ...
        params.theta*hprime(W(:,k)'*(X(:,k)-X(:,j)))*params.topfactor*(X(:,k)-X(:,j));
  obj = obj + 2*params.lambda2*W(:,k);
end


function y=h(x)
y = max(1-x,0.0);

function y=hprime(x)
y = (x*0-1);
y(x>=1) = 0;


