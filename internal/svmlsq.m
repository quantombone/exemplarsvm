function [w,svmobj] = svmlsq(y,x,lambda,w,NITER,params)
%Compute optimal solution for hinge-squared SVM problem
%By guessing the support vectors, solving a least-squares
%optimization problem, then iterating between updating the support
%vectors and re-solving the LSQ problem.  Each time a new solution is
%obtained, a coarse line search is performed to guarantee that the SVM
%objective function does not increase.
% NOTE: works for L2-loss svms and makes sure bias is not
% regularized
%
% Input
%  y: vector of classes
%  x: data matrix
%  lambda: regulariation parameters
%  w: current estimate of w
%
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

%Maximum number of newton iterations
if ~exist('NITER','var')
  NITER = 20;
end

%Use liblinear if number of iterations is negative
if NITER < 0
  [w,svmobj] = learn_ll(x,y,lambda,1);
  return;
end

lambdaI = lambda*eye(size(x,1));
if exist('params','var') && isfield(params,'regularizer')
  lambdaI = params.regularizer;
else
  params.basis = eye(size(x,1));
  params.regularizer = eye(size(x,1));
  params.display = 1;
end

F = size(x,1);
if ~exist('w','var')
  w = zeros(F+1,1);
end
oldw = w;
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
    goods = 1:length(y);
  else
    r = (y'.*(w(1:end-1)'*x+w(end)));
    goods = find(r<=1.0);
  end

  newgoods = setdiff(goods,oldgoods);
  oldgoods = setdiff(oldgoods,goods);
  curmat = curmat + x(:,newgoods)*x(:,newgoods)' - x(:,oldgoods)*x(:,oldgoods)';
  
  sx = sum(x(:,goods),2);
  M = [lambdaI + 2*curmat 2*sx;...
       sx' length(goods)];
  U = [2*x(:,goods)*y(goods); sum(y(goods))];

  w = M\U;  
  
  [w,bestobj] = line_search(w,oldw,y,x,lambda,params,linspace(0,1,10));

  num_nsv = sum( (w(1:end-1)'*x(:,y==-1) + w(end)) >= -1);
  
  endtime = toc(starttime);
  if params.display == 1
    fprintf(1,' ---curobj=%.3f (iter in %.3f s, +s: %d -s: %d #neg-sv %d)\n',...
            bestobj,endtime,length(newgoods),length(oldgoods), ...
            num_nsv);
  end

  if (isempty(newgoods) && isempty(oldgoods))
    %fprintf(1,'breaking because new objective is %.3f\n',bestobj);
    break;
  end
  
  oldw = w;  
  oldgoods = goods;
end

if nargout == 2
  svmobj = bestobj;
  if params.display == 1
    svmobj =  lambda/2*sum((params.basis*w(1:end-1)).^2) + sum(hinge(y'.*(w(1:end-1)'*x+w(end))));
    fprintf(1,'curobj=%.3f\n',svmobj);
  end
end


function [w,bestobj] = line_search(w,oldw,y,x,lambda,params,alphas)
%Perform the line search between solutions w and oldw by looking at
%solutions as follows: (alpha)*w + (1-alpha)*oldw
if ~exist('alphas','var')
  alphas = linspace(0,1,10);
end
bestw = w;
bestobj = inf;
r1 = y'.*(w(1:end-1)'*x + w(end));
r2 = y'.*(oldw(1:end-1)'*x + oldw(end));
n1 = sum((params.basis*w(1:end-1)).^2, 1);
n2 = sum((params.basis*oldw(1:end-1)).^2, 1);
ip = w(1:end-1)'*params.regularizer*oldw(1:end-1);
newobj = zeros(length(alphas),1);
for q = 1:length(alphas)
  alpha = alphas(q);
  
  newobj(q) = lambda/2*(alpha^2*n1+...
      (1-alpha).^2*n2 + ...
      2*alpha*(1-alpha)*ip) + ...
      sum(hinge(r1*alpha+(1-alpha)*r2));
  
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
