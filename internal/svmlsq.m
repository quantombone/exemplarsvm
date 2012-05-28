function [w,svmobj] = svmlsq(y,x,w,params)
%Compute optimal solution for hinge-squared SVM problem
%By guessing the support vectors, solving a least-squares
%optimization problem, then iterating between updating the support
%vectors and re-solving the LSQ problem.  Each time a new solution is
%obtained, a coarse line search is performed to guarantee that the SVM
%objective function does not increase.
% NOTE: works for L2-loss svms and makes sure bias is not
% regularized
% Minimizes the following objective
% objective(w) = w'*R*w + w'*c + sum_{i=1}^N max(1-y_i*w'*x_i,0)^2
%
% Input
%  y: vector of classes
%  x: data matrix
%  w: current estimate of w
%  params: coefficients and such such 
%
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

%Maximum number of newton iterations
if ~isfield(params,'NITER')
  params.NITER = 20;
end
if ~isfield(params,'display')
  params.display = 0;
end

if ~isfield(params,'e')
  params.e = zeros(size(y))';
end

if ~isfield(params,'weights')
  params.weights = ones(size(y'));
end

%Use liblinear if number of iterations is negative
if params.NITER < 0
  [w] = learn_ll(x,y,params.regularizer(1),1);
  

  svmobj = w'*params.regularizer*w + sum(hinge(y'.*(w(1:end-1)'*x+w(end))));
  
  return;
end

%lambdaI = params.regularizer;
%lambdaI = lambda*params.regularizer;
%lambdaI = lambda*eye(size(x,1));
%if exist('params','var') && isfield(params,'regularizer')
%  lambdaI = lambda*params.regularizer;
%else
%  params.basis = eye(size(x,1));
%  params.regularizer = eye(size(x,1));
%  params.display = 1;
%end

F = size(x,1);
if ~exist('w','var')
  w = zeros(F+1,1);
end
oldw = w;
oldgoods = [];
curmat = zeros(F,F);


oldobj = w'*params.regularizer*w + w'*params.c + ...
         sum((params.weights(:)').*hinge(y'.*(w(1:end-1)'*x+w(end)+params.e)));

if params.display == 1
  fprintf(1,'-++curobj=%.3f\n',oldobj)
end

for i = 1:params.NITER
  starttime=tic;
  if (i == 1) && (~exist('w','var') || sum(abs(w(:)))==0)
    goods = 1:length(y);
  else
    r = (y'.*(w(1:end-1)'*x+w(end)+params.e));
    goods = find(r<=1.0);
    if length(goods) <= 10
      fprintf(1,'Warning, fewer than 10 SVs\n');
      %goods = 1:length(y);
      %There can be a problem if initial solution only hits one
      %points, so we take the two points with
      [alpha,beta] = sort(r,'ascend');
      %beta1 = beta(y(beta)>0);
      %beta2 = beta(y(beta)<0);
      %goods = [beta1(1) beta2(1)];
      goods = beta(1:min(length(y),20));
    end
    
  end

  newgoods = setdiff(goods,oldgoods);
  oldgoods = setdiff(oldgoods,goods);
  curmat = curmat + x(:,newgoods)*x(:,newgoods)' - x(:,oldgoods)*x(:,oldgoods)';
  
  sx = sum(x(:,goods),2);
  % M = [params.regularizer(1:end-1,1:end-1)+2*curmat 2*sx;...
  %      sx' length(goods)];
  % U = [2*x(:,goods)*y(goods); sum(y(goods))];
  
  M = 2*params.regularizer+[2*curmat 2*sx;...
                    2*sx' 2*length(goods)];
  U = [2*x(:,goods)*y(goods); 2*sum(y(goods))]-params.c;


  w = M\U;  
  
  [w,bestobj,bestalpha] = line_search(w,oldw,y,x,params,linspace(0, ...
                                                  1,10));

  
  svmobj = w'*params.regularizer*w + w'*params.c + sum(hinge(y'.*(w(1:end-1)'*x+w(end))));

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
    svmobj = w'*params.regularizer*w + w'*params.c + sum(hinge(y'.*(w(1:end-1)'*x+w(end))));
    fprintf(1,'curobj=%.3f\n',svmobj);
  end
end


function [w,bestobj,bestalpha] = line_search(w,oldw,y,x,params,alphas)
%Perform the line search between solutions w and oldw by looking at
%solutions as follows: (alpha)*w + (1-alpha)*oldw
if ~exist('alphas','var')
  alphas = linspace(0,1,10);
end
bestw = w;
bestobj = inf;
bestalpha = inf;
r1 = y'.*(w(1:end-1)'*x + w(end));
r2 = y'.*(oldw(1:end-1)'*x + oldw(end));
n1 = w'*params.regularizer*w;
n2 = oldw'*params.regularizer*oldw;

c1 = w'*params.c;
c2 = oldw'*params.c;
%n1 = sum((params.basis*w(1:end-1)).^2, 1);
%n2 = sum((params.basis*oldw(1:end-1)).^2, 1);
ip = w'*params.regularizer*oldw;
%ip = w(1:end-1)'*params.regularizer*oldw(1:end-1);
newobj = zeros(length(alphas),1);
for q = 1:length(alphas)
  alpha = alphas(q);
  
  newobj(q) = (alpha^2*n1 + ...
      (1-alpha).^2*n2 + ...
      2*alpha*(1-alpha)*ip) + ...
      (alpha*c1 + (1-alpha)*c2) + ...
      sum(hinge(r1*alpha + (1-alpha)*r2));
  
  if (newobj(q) < bestobj)
    bestw = alpha*w+(1-alpha)*oldw;
    bestobj = newobj(q);
    bestalpha = alpha;
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

