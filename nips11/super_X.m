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
%targety = [111 145 198];
%targety = [111 134];

%targety = [111 359 383:400];
%targety = [111 257 259];
%targety = 1:10;
%targety = [259 388 43];

goods = find(ismember(y,targety));
others = find(~ismember(y,targety));
r = randperm(length(others));
others = others(1:2000);

ids = ids([goods; others]);
X = X(:,[goods; others]);
y = y([goods; others]);

if 0
  for k = 1:length(y)
    fprintf(1,'.');
    [xs{k},bbs,Is{k}] = get_global_wiggles(convert_to_I(ids{k}));
    ys{k} = ones(size(xs{k},2),1)*y(k);
  end
  y = cat(1,ys{:});
  X = cat(2,xs{:});
  ids=cat(1,Is{:});
end

params.learning_rate = .01;

%capacity gain coefficient
params.gamma = 0;

%how much to separate positives and negatives
params.lambda = 100;

%how much to regularize the norms of w 
params.lambda2 = .1;

%how much to force nearby w to be close on L2 sense
params.sigma = 0; %1;%.01; %.01;

%how much to force max-self constraint
params.theta = 0;
params.topfactor = .1;

W = X(:,1:length(goods));
mx = mean(X,2);
for i = 1:size(W,2)
  W(:,i) = X(:,i);
  W(:,i) = W(:,i) - mean(W(:,i));
end
%W = W*0;

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
X2(end+1,:) = 1;
W(end+1,:) = 0;

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

  if mod(q,20)==0
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
%evaluate the objective function, slow but handy

N = size(X,2);
obj = zeros(size(W,1),1);

%r = randperm(N);
for j = [ceil(rand*N)] %1:N
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


function [W,A,iter] = update_stuff(W,A,params,X,y,iter)

%gain matrix is identity
G = ones(size(A));

for chunks = 1:5
  
  %obj = evaluate_objective(W,A,params,X);
  %grad = evaluate_gradient(W,A,X,1);    
  r = [randperm(size(W,2))];% randperm(size(W,2));];
  %r = r(1:100);
  %[aa,bb] = sort(W(:,16)'*X,'descend');
  %r = [16 bb(1:50)];
  %r = ones(100,1)*16;
  %r = 1:50;
  %r = repmat(r,100,1);
  
  %r = r(1:min(length(r),2000));
  
  %r = 1;
  %r = r(1);
  %r = ones(100,1)*2;
  
  for z = 1:length(r)
    grad = evaluate_gradient(W,A,params,X,r(z),G);
    W(:,r(z)) = W(:,r(z)) - params.learning_rate/sqrt(iter+1)*grad;
    iter = iter + 1;
  end  
end
return;
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
