function [w,svmobj] = svmlsq(y,x,lambda,w,NITER,params)
%Compute optimal solution for hinge-squared SVM problem
%By guessing the support vectors, solving a least-squares
%optimization problem, then iterating between updating the support
%vectors and re-solving LSQ
%NOTE: works for L2-loss svms and makes sure bias is not
%regularized
% Input
%  y: vector of classes
%  x: data matrix
%  lambda: regulariation parameters
%  w: current estimate of w
%Tomasz Malisiewicz (tomasz@csail.mit.edu)


%maximum number of newton iterations
if ~exist('NITER','var')
  NITER = 20;
end

if NITER == -1
  [w,svmobj] = learn_ll(x,y,lambda,1);
  return;
end

lambdaI = lambda*eye(size(x,1));
if exist('params','var') && isfield(params,'regularizer')
  lambdaI = lambda*params.regularizer;
else
  params.basis = eye(size(x,1));
  params.regularizer = lambda*eye(size(x,1));
  params.display = 1;
end

%b = ones(1,size(x,1));
%b(end) = 0;
%BtB = lambda*diag(b.^2);
%BtBinv = 1/lambda*diag(1./(b.^2));
%BtBinv(end) = 0;

F = size(x,1);
if ~exist('w','var')
  w = zeros(F+1,1);
end
oldw = w;
nostart = (sum(w(1:end-1).^2)==0);
oldgoods = [];
curmat = zeros(F,F);

oldobj =  lambda/2*sum((params.basis*w(1:end-1)).^2) + ...
          sum(hinge(y'.*(w(1:end-1)'*x+w(end))));
if params.display == 1
  fprintf(1,' -++curobj=%.3f\n',oldobj)
end

for i = 1:NITER
  starttime=tic;
  if (i == 1) && (~exist('w','var') || sum(abs(w(:)))==0)
    %goods = randperm(length(y));
    %goods = goods(1:min(length(goods),100));
    %goods = unique([goods'; find(y==1)]);
    goods = 1:length(y);
  else
    r = (y'.*(w(1:end-1)'*x+w(end)));

    % [aa,bb] = sort(r,'ascend');
    % aa = aa(1:i*1000);
    % bb = bb(1:i*1000);
    % goods = bb(aa<=1);
    

    %choose all of them
    goods = find(r<=1.0);
  end


  %curmat = x(:,goods)*x(:,goods)';
  newgoods = setdiff(goods,oldgoods);
  oldgoods = setdiff(oldgoods,goods);
  curmat = curmat + x(:,newgoods)*x(:,newgoods)' - x(:,oldgoods)*x(:,oldgoods)';
  %M = (BtB+2*curmat);
  %U = 2*x(:,goods)*y(goods);
  
  sx = sum(x(:,goods),2);
  M = [lambdaI + 2*curmat 2*sx;...
       sx' length(goods)];
  U = [2*x(:,goods)*y(goods); sum(y(goods))];

  % if i <= 3
  %   grad = M*w+U;
  %   w = w + .00001*grad;
  % else
  %w = pinv(M)*U;
  w = M\U;  
  %end
  
  [w2,bestobj] = line_search(w,oldw,y,x,lambda,params,linspace(0,1, ...
                                                  10));

  num_nsv = sum( (w(1:end-1)'*x(:,y==-1) + w(end)) >= -1);
  
  %svmobj =  lambda/2*sum(w(1:end-1).^2) + sum(hinge(y'.*(w'*x)));
  endtime = toc(starttime);
  if params.display == 1
    fprintf(1,' ---curobj=%.3f (iter in %.3f s, +s: %d -s: %d #neg-sv %d)\n',...
            bestobj,endtime,length(newgoods),length(oldgoods), ...
            num_nsv);
  end

  if (length(newgoods) == 0 && length(oldgoods)==0)%(oldobj - bestobj)/oldobj < .001
    %fprintf(1,'breaking because new objective is %.3f\n',bestobj);
    break;
  end
  oldw = w;  

  oldobj = bestobj;
  oldgoods = goods;
end

if nargout == 2
  svmobj = bestobj;
  if params.display == 1
    svmobj =  lambda/2*sum((params.basis*w(1:end-1)).^2) + sum(hinge(y'.*(w(1:end-1)'*x+w(end))));
    fprintf(1,'curobj=%.3f\n',svmobj);
  end
end


function [gw] = compute_gradient(y,x,w,lambda)

%Compute the w-norm part
gw = lambda*w;
gw(end) = 0;

%compute the w on positives part
r = y'.*(w'*x);
u = find(r<1);

for i = 1:length(u)
  gw = gw + hingeprime(r(u(i)))*y(u(i))*x(:,u(i));
end

function [w,bestobj] = line_search(w,oldw,y,x,lambda,params,alphas)
%perform the line search  
if ~exist('alphas','var')
  alphas = linspace(0,1,10);
end
bestw = w;
bestobj = 100000000;
r1 = y'.*(w(1:end-1)'*x+w(end));
r2 = y'.*(oldw(1:end-1)'*x+oldw(end));
n1 = sum((params.basis*w(1:end-1)).^2);
n2 = sum((params.basis*oldw(1:end-1)).^2);

ip = w(1:end-1)'*params.regularizer*oldw(1:end-1);

for q = 1:length(alphas)
  alpha = alphas(q);
  
  newobj(q) = lambda/2*(alpha^2*n1+...
      (1-alpha).^2*n2+...
      2*alpha*(1-alpha)*ip) ...
      + sum(hinge(r1*alpha+(1-alpha)*r2));
  
  if (newobj(q) < bestobj)
    bestw = alpha*w+(1-alpha)*oldw;
    bestobj = newobj(q);
  end
end

w = bestw;

function [w,svmobj,t] = learn_ll(x,y,lambda,meancoeff)
starttime = tic;

if ~exist('meancoeff','var')
  meancoeff = 0;
end

params.bias = 100;
mx = mean(x,2)*meancoeff;
x2 = bsxfun(@minus,x,mx);
svm_model = liblineartrain(y, sparse(x2)',sprintf(['-s 2 -B %.5f -e .001 -c' ...
                    ' %f -q'], params.bias, 1/lambda));
w = svm_model.w';
w(end) = (w(end)*params.bias);
t = toc(starttime);

w(end) = w(end) - w(1:end-1)'*mx;

vals = y'.*(w(1:end-1)'*x+w(end));
svmobj =  lambda/2*sum(w(1:end-1).^2) + sum(hinge(vals));
return;
