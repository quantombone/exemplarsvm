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
targety = [259 255 307 112 32 44 56];
%targety = 1:10;
%targety = [259 388];
[aa,bb] = ismember(y,targety);
goods = find(aa==1);
others = find(y<=0);
finaly = y([goods; others]);

ids = ids([goods; others]);
X2 = X(:,others);
X = X(:,goods);
X = cat(2,X,X2);
y = y([goods; others]);

params.learning_rate = .01;
params.gamma = 1;
params.lambda = 4;
params.sigma = 1.0;
params.theta = 10.0;
params.topfactor = 1;



W = X;
mx = mean(X,2);
for i = 1:size(W,2)
  W(:,i) =(W(:,i) - mx);
end


N = size(X,2);
A = eye(N,N);

%d = distSqr_fast(X,X);
%A = (d<.5*median(d(:)));
%A(find(speye(size(A))))=1;


if 0
  cx = repmat(finaly(:),1,length(finaly));
  cy = cx';
  A = A*0;
  A(cx==cy) = 1;
end



shower = zeros(100,100,3);
X(end+1,:) = 1;
W(end+1,:) = 0;

for q = 1:2000


  if mod(q,100) == 0
    fprintf(1,',');
  end
  
  %obj = evaluate_objective(W,A,params,X);
  %grad = evaluate_gradient(W,A,X,1);
  
  
  r = [randperm(size(W,2)) randperm(size(W,2));];
  
  %r = r(1:min(length(r),2000));
    
  %r = r(1);
  for z = 1:length(r)
    grad = evaluate_gradient(W,A,params,X,r(z));
    W(:,r(z)) = W(:,r(z)) - params.learning_rate*grad;
  end
  
  
  
  
% [aa,bb] = sort(A(1,:),'descend');

% F = size(X,1);
% lambda = 10;
% tic
% W = pinv((X*X'+lambda*eye(F))')*X*A';
% %K = X'*X;
% %E = inv((K'*K+lambda*K')')*A*K';
% %W = X*E;
% toc

% A = W'*X;


% for i = 1:size(A,1)
%   [aa1,bb1] = sort(A(i,:),'descend');
%   %A(i,:) = 0;
%   [aaa,bbb]= sort(bb1);
%   %A(i,:) = 1./(bbb);
% end

%never change A
if q > 10
  hmat = h(W'*X);
  A = double(hmat+hmat'<2*params.gamma);
  A(find(speye(size(A))))=1;
  
  r = find(randn<.01);
  A(r) = 1-A(r);
  A = (A+A')/2;
  
  
end

figure(1)
clf

if mod(q,100)==0
  %shower = create_grid_res(ids(bb),4);
end

clf
subplot(1,3,1)

%imagesc(shower)


%keyboard
%imagesc(d)
imagesc(W'*X)

subplot(1,3,2)
%plot(W(:,2)'*X)

r = W'*X;
plot(diag(r))
subplot(1,3,3)

fprintf(1,'showing memex graph\n');
if q < 1000
  imagesc(A .*( W'*X))
else
  tic
  I = show_memex(A,ids,y);
  toc
  imagesc(I)
end
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
end

function obj = evaluate_gradient(W,A,params,X,k)
%evaluate the objective function, slow but handy

%gain matrix is identity
G = ones(size(A));

N = size(X,2);
obj = zeros(size(X,1),1);

for j = ceil(rand*N) %1:N
  if A(k,j) ~= 0
    obj = obj + params.sigma*2*(W(:,k)-W(:,j));
    
    obj = obj + params.lambda*(hprime(W(:,k)'*X(:,j))*X(:,j)-hprime(-W(:,k)'*X(:,j))*-X(:,j));
  end
  
  obj = obj + params.lambda*hprime(-W(:,k)'*X(:,j))*-X(:,j) + params.theta*hprime(W(:,k)'*(X(:,k)-X(:,j)))*params.topfactor*(X(:,k)-X(:,j));
end


function y=h(x)
y = max(1-x,0.0);

function y=hprime(x)
y = (x*0-1);
y(x>=1) = 0;


