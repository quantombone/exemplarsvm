function super_X(X,ids,y,X2,ids2,y2,catnames)
%% Here we learn a discriminative manifold for a set of X's

if ~exist('ids','var')
  [X,ids,y,catnames] = sun_initialize('Training_01');
end

if ~exist('ids2','var')
  %Load the testset
  [X2,ids2,y2,catnames] = sun_initialize('Testing_01');
end

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

%allX = X;
%allX(end+1,:) = 1;
%ally = y;
%allids = ids;

targety = 1;

%choose phone booths and control towers
targety = [259 255 388 307 1:10];
%targety = [259 255 307 112 32 44 56];
targety = [111 123 148 213 217 220 230 240 250];
targety = [275 278];
targety = [111];
targety = 111;
targety = 275;
%targety = 245;
%targety = [111 145 198];
%targety = [111 134];

%targety = [111 359 383:400];
%targety = [111 257 259];
%targety = 1:10;
%targety = [259 388 43];

goods = find(ismember(y,targety));
%goods = [];
others = find(~ismember(y,targety));
r = randperm(length(others));
others = others(1:2000);

%idssave = ids(goods2);
ids = ids([goods; others]);
X = X(:,[goods; others]);
y = y([goods; others]);

if 0
  for k = 1:length(goods2)
    fprintf(1,'.');
    [xs{k},bbs,Is{k}] = get_global_wiggles(convert_to_I(idssave{k}));
    ys{k} = ones(size(xs{k},2),1)*y(k);
  end
  yA = cat(1,ys{:});
  XA = cat(2,xs{:});
  idsA=cat(1,Is{:});

  X = cat(2,XA,X);
  y = cat(1,yA,y);
  ids = cat(1,idsA,ids);
  goods = 1:length(goods2);
  
end

params.learning_rate = 1;

%capacity gain coefficient
params.gamma = 0;

%how much to separate positives and negatives
%params.lambda = 10;

%how much to regularize the norms of w 
params.lambda = .01;%.01;

%how much to force nearby w to be close on L2 sense
params.sigma = .00; %1;%.01; %.01;

%how much to force max-self constraint
params.theta = 0;
%params.topfactor = 1;

W = X(:,1:length(goods));
mx = mean(X,2);
for i = 1:size(W,2)
  W(:,i) = X(:,i);
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
A(cx~=cy) = -1;
A(cx==cy) = 1;
%A = Asave;

shower = zeros(100,100,3);

W(end+1,:) = 0;

X(end+1,:) = 1;
X2(end+1,:) = 1;

iter = 1;
for q = 1:2000
  if mod(q,100) == 0
    fprintf(1,',');
  end
  

  [W,A,iter] = update_stuff(W,A,params,X,y,iter);
  
  %A = eye(size(A));
  %A = Asave;
  
  figure(1)
  if mod(q,100)==0
    %shower = create_grid_res(ids(bb),4);
  end
  
  subplot(2,2,1)
  imagesc(W'*X)
  title(sprintf('W''X Matrix iter %d',q))
  
  subplot(2,2,2)
  r = W'*X;
  sigmoid = @(x)(1./(1+exp(-x)));
  imagesc(sigmoid(.2*r(:,1:200)))
  title('matrix after sigmoids')
  
  subplot(2,2,3)
  
  if 1 %q < 1000
    kkk = size(W,2);
    imagesc(A(1:kkk,1:kkk))
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
  
  hhh = W'*X;
  plot(hhh(16,:),'r.');
  
  %kkk = min(size(hhh));
  %ddd = diag(hhh(1:kkk,1:kkk));
  %[aaa,bbb] = max(hhh(1:kkk,1:kkk),[],1);
  %plot(ddd-aaa');
  %drawnow
  drawnow

  if mod(q,10)==0
    %r = W(:,1:50)'*X2;
    %r = m(r,[],1);
    %r = sum(sigmoid(.2*r),1);
    
    figure(2)
    %target = sum(A,1);
    %[aa,target] = max(target);
    
    %[aa,targets] = sort(diag(W'*X),'descend');
    target = 16;
    r = W(:,target)'*X2;
    [aa,bb] = sort(r,'descend');
    hw = zeros(10,10,31);
    hw(2:9,2:9,:) = reshape(W(1:end-1,target),[8 8 31]);
    
    hogger = HOGpicture(hw);
    newids{1} = ids{target};
    newids{2} = hogger;
    newids(3:16) = ids2(bb(1:14));

    images = create_grid_res(newids,4);
    
    %images(3:end+1) = images(2:end);
    %images{2} = hogger;
    imagesc(images)
    drawnow
  end
  
  %fprintf(1,'capping to asave\n');
  %A = A.*Asave;
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

function obj = evaluate_gradient(W,A,params,X,k,G)
%evaluate the stochastic gradient for weight vector k with respect
%to a random sample j

N = size(X,2);
obj = zeros(size(W,1),1);

% for j = [ceil(rand*N)
%   if A(k,j) ~= 0
%     obj = obj + params.sigma*2*(W(:,k)-W(:,j));
    
%     obj = obj + params.lambda*(hprime(W(:,k)'*X(:,j))*X(:,j)-hprime(-W(:,k)'*X(:,j))*-X(:,j));
%   end
  
%   obj = obj + params.lambda*hprime(-W(:,k)'*X(:,j))*-X(:,j) + ...
%         params.theta*hprime(W(:,k)'*(X(:,k)-X(:,j)))*params.topfactor*(X(:,k)-X(:,j));
%   obj = obj + 2*params.lambda2*W(:,k);
% end

%j = [ceil(rand*N)];
[aa,j] = max(h(A(k,:).*(W(:,k)'*X)));


obj = 2*params.lambda*W(:,k);

if k ~= j
  obj = obj + params.theta * hprime(W(:,k)'*(X(:,k)-X(:,j)))*(X(:,k)-X(:,j));
end

if A(k,j) ~= 0
  obj = obj + hprime(A(k,j)*W(:,k)'*X(:,j))*A(k,j)*X(:,j);
  if 0
  masker = reshape(1:64,[8 8]);
  masker = repmat(masker,[1 1 31]);
  for q = 1:max(masker(:))
    subby = [find(masker(:)==q); size(X,1)];
    obj(subby) = obj(subby) + .1*hprime(A(k,j)*W(subby,k)'*X(subby,j))*A(k,j)*X(subby,j);
  end
  end
end

if A(k,j) == 1 && j <= size(W,2)
  obj = obj + params.sigma*2*(W(:,k)-W(:,j));
end


function y=h(x)
y = max(1-x,0.0);

function y=hprime(x)
y = (x*0-1);
y(x>=1) = 0;


function [W,A,iter] = update_stuff(W,A,params,X,y,iter)

%gain matrix is identity
G = ones(size(A));

for chunks = 1:5
  
  %obj = evaluate_objective(W,A,params,X);
  %grad = evaluate_gradient(W,A,X,1);    
  r = [randperm(size(W,2))];
  %r = r(1:100);
  %[aa,bb] = sort(W(:,16)'*X,'descend');
  %r = [16 bb(1:50)];
  %r = ones(100,1)*16;
  %r = 1:50;
  
  %r = repmat(r,1,10);

  
  %r = r(1:min(length(r),2000));
  %r = 1;
  %r = r(1);
  %r = ones(100,1)*16;

  %r = 16;
  for z = 1:length(r)
    grad = evaluate_gradient(W,A,params,X,r(z),G);
    W(:,r(z)) = W(:,r(z)) - params.learning_rate/sqrt(iter+1)*grad;
    iter = iter + 1;
  end  
end

%no A update


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
  for a = 1:size(W,2)
    cur = W(:,a)'*X;
    gains = double(y==y(a));
    gains(gains==0) = -1;
    [aa,bb] = sort(cur,'descend');
    
    cs = cumsum(gains(bb));
    [alpha,beta] = max(cs);
    if norm(W(:,a))==0
      beta = [];
    end
    
    %beta = [beta];
    %purity = mean(y(bb(1:beta)) == y(q));
    %fprintf(1,'beta is %d purity is %.3f\n',beta,purity);
    
    A(a,bb(1:beta)) = 1;
    A(a,a) = 1;
  end
  A = double(A&A');  
end
