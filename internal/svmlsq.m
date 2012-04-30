function [w,svmobj] = svmlsq(y,x,lambda,w)
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
NITER = 20;

b = ones(1,size(x,1));
b(end) = 0;
BtB = lambda*diag(b.^2);
%BtBinv = 1/lambda*diag(1./(b.^2));
%BtBinv(end) = 0;

F = size(x,1);
if ~exist('w','var')
  w = x(:,1)*0;
end
oldw = w;
nostart = (sum(w(1:end-1).^2)==0);
oldgoods = [];
curmat = zeros(F,F);

oldr = (y'.*(w'*x));

svmobj =  lambda/2*sum(w(1:end-1).^2) + sum(hinge(y'.*(w'*x)));
fprintf(1,' -++curobj=%.3f\n',svmobj)


for i = 1:NITER
  starttime=tic;
  if (i == 1) && (~exist('w','var') || sum(abs(w(:)))==0)
    %goods = randperm(length(y));
    %goods = goods(1:min(length(goods),100));
    %goods = unique([goods'; find(y==1)]);
    goods = 1:length(y);
  else
    r = (y'.*(w'*x));

    % [aa,bb] = sort(r,'ascend');
    % aa = aa(1:i*500);
    % bb = bb(1:i*500);
    % goods = bb(aa<=1);

    %choose all of them
    goods = find(r<=1.0);
  end

  newgoods = setdiff(goods,oldgoods);
  oldgoods = setdiff(oldgoods,goods);
  curmat = curmat + x(:,newgoods)*x(:,newgoods)' - x(:,oldgoods)*x(:,oldgoods)';
  
  M = (BtB+2*curmat);
  U = 2*x(:,goods)*y(goods);
  w = M\U;  
  
  %perform the line search  
  alphas = linspace(0,1,10);
  bestw = w;
  bestobj = 100000000;
  for q = 1:length(alphas)
    alpha = alphas(q);
    w2 = alpha*w+(1-alpha)*oldw;
    newobj(q) =  lambda/2*sum(w2(1:end-1).^2) + sum(hinge(y'.*(w2'* ...
                                                  x)));
    
    if (newobj(q) < bestobj)
      bestw = w2;
      bestobj = newobj(q);
    end
  end
  w = bestw;
    
  if norm(w-oldw)<.01
    break;
  end
  oldw = w;  
  svmobj =  lambda/2*sum(w(1:end-1).^2) + sum(hinge(y'.*(w'*x)));
  endtime = toc(starttime);
  fprintf(1,' ---curobj=%.3f (iter in %.3f s, ngoods=%d)\n',svmobj,endtime,length(newgoods));

  oldgoods = goods;
end

if nargout == 2
  svmobj =  lambda/2*sum(w(1:end-1).^2) + sum(hinge(y'.*(w'*x)));
  fprintf(1,'curobj=%.3f\n',svmobj);
end

