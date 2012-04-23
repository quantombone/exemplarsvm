function [w,b,obj] = primal_svm(Y,lambda,X,wold,bold,opt)
% [w, b, obj] = PRIMAL_SVM(Y,LAMBDA,X,wstart)
% Solves the SVM optimization problem in the primal (with quadratic
%   penalization of the training errors).  
%
% LAMBDA is the regularization parameter ( = 1/C)
% 
% Copyright Olivier Chapelle, olivier.chapelle@tuebingen.mpg.de
% Last modified 25/08/2006  

% Modifications by Tomasz Malisiewicz (tomasz@cmu.edu)
% The objective function that is being solved is as follows:
% f(w) = lambda*||w||^2 / 2 + ...
%        sum_{i=1}^{N} max(0,1-y_i*(w'*x_i + b))^2

%opt.iter_max_Newton = 20;
%opt.prec = 1e-6;

if ~exist('opt','var')
  opt.iter_max_Newton = 50;
  opt.prec = 1e-6;
end
[n,d] = size(X);

wstart = zeros(d+1,1);
if exist('wold','var')
  wstart(1:(end-1)) = wold;
  wstart(end) = bold;
end

if size(Y,1)~=n, Y=Y'; end;

if size(Y,1)~=n, error('primal_svm: Dimension error'); end;
%[solution,obj,out] = primal_svm_linear(Y,lambda*n,X,opt,wstart);
%obj = obj / n;

[solution,obj,out] = primal_svm_linear(Y,lambda,X,opt,wstart);

% The last component of the solution is the bias b.
b = solution(end);
w = solution(1:end-1);

%score = sum( max(1-Y.*(X*w+b),0).^2)/2 + lambda*norm(solution).^2/2;
%keyboard

function  [w,obj,out] = primal_svm_linear(Y,lambda,X,opt,wstart) 
% -------------------------------
% Train a linear SVM using Newton 
% -------------------------------
[n,d] = size(X);

w = wstart;

hess_constant = lambda*diag([ones(d,1); 0]);

iter = 0;
out = 1-Y.*(X*w(1:end-1) + w(end));

while 1
  iter = iter + 1;
  fprintf(1,'.');
  if iter > opt.iter_max_Newton;
    %warning(sprintf(['Maximum number of Newton steps reached.' ...
    %                 'Try larger lambda']));
    break;
  end;
  
  [obj, grad, sv, Xsv] = obj_fun_linear(w,Y,lambda,X,out);
  %w = w - .00001*grad;
  %continue
  % Compute the Newton direction
  %Xsv = X(sv,:);
  %fprintf(1,'size X: %d size Xsv: %d\n',size(X,1),size(Xsv,1));
  %Hessian

  if 1
    
    
     hess = hess_constant + ...
            [[Xsv'*Xsv sum(Xsv,1)']; [sum(Xsv,1) length(sv)]];
    

    %hess = hess_constant;
    
    step  = - hess \ grad;   
  else
    step = -grad;
  end

  % Do an exact line search
  [t,out] = line_search_linear(w,step,out,Y,X,lambda);

  if t <= 0 | -step'*grad < opt.prec * obj(1) 
    %fprintf(1,'stopping since newton dec small %f %f\n',-step'*grad,opt.prec*obj);
    % Stop when the Newton decrement is small enough
    break;
  end;

  w = w + t*step;
  
  %fprintf(['Iter = %.2d, Obj = %.6f, #SV = %d, Newton decr = %.3f, ' ...
  %         't = %.9f\n'],iter,obj(1),length(sv),-step'*grad,t);

end;

function [obj, grad, sv, Xsv] = obj_fun_linear(w,Y,lambda,X,out)
% Compute the objective function, its gradient and the set of support vectors
% Out is supposed to contain 1-Y.*(X*w)
out = max(0,out);
sv = logical(out>0);
Xsv = X(sv,:);
w0 = w; w0(end) = 0;  % Do not penalize b
obj(1) = sum(out(sv).^2)/2 + lambda*(w0'*w0)/2;
obj(2) = sum((out(sv).*(Y(sv)==-1)).^2)/2;
obj(3) = sum((out(sv).*(Y(sv)==1)).^2)/2;
obj(4) = lambda*(w0'*w0)/2;
grad = lambda*w0 - [((out(sv).*Y(sv))'*Xsv)'; sum(out(sv).*Y(sv))]; % Gradient

function [t,out] = line_search_linear(w,d,out,Y,X,lambda) 
% From the current solution w, do a line search in the direction d by
% 1D Newton minimization

t = 0;
% Precompute some dots products
Xd = X*d(1:end-1)+d(end);
wd = lambda * w(1:end-1)'*d(1:end-1);
dd = lambda * d(1:end-1)'*d(1:end-1);
maxiter = 10;
numiter = 0;

%HACK to always have t=1
%this also works just as well it seems
%t = 1;
%out = out - t*(Y.*Xd);
%return;

%another little speed-up here
YXd = Y.*Xd;
while 1
  %out2 = out - t*(Y.*Xd); % The new outputs after a step of length t
  out2 = out - t*(YXd); % The new outputs after a step of length t
  sv = logical(out2>0);
  g = wd + t*dd - (out2(sv).*Y(sv))'*Xd(sv); % The gradient (along the line)
  h = dd + Xd(sv)'*Xd(sv); % The second derivative (along the
                               % line)

  %fprintf('%f %f %f %f\n',t,g^2/(h+eps), g^2,h)
  if (g^2/(h+eps) < 1e-10) | (numiter>maxiter) | abs(g/(h+eps)) < 1e-5
  %if (abs(g) < 1e-8) | (numiter>maxiter)
    out = out2;
    break; 
  end;  
                           
  t = t - g/h; % Take the 1D Newton step. Note that if d was an exact Newton
               % direction, t is 1 after the first iteration.
  %fprintf(1,'t is %.5f g is %.5f h is %.5f\n',t,g,h);
  

  numiter = numiter+1;
end

out = out2;
