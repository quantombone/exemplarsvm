function demo_regtrain
D = 2;
N = 4000;
KREG = 10;

superx = randn(D,N);
supery = double(rand(N,1)>.5);
supery(1) = 1;
supery(supery==0) = -1;
superx(:,supery==1) = superx(:,supery==1) + 2.0;
if 0
figure(1)
for i = 1:10
  tic
  [w2,b2] = reg(superx,supery,i);
  toc
  clf
  plot_them(superx,supery,w2,b2)
  title(num2str(i))
  drawnow
  pause
end
return;
end

starter = tic;
[w2,b2,r2] = reg(superx,supery,KREG);
fprintf(1,'reg time %.5f sec\n',toc(starter));

starter = tic;
[w,b,r] = svm(superx,supery);
fprintf(1,'svm time %.5f sec\n',toc(starter));

plot(r,r2,'r.');
return;

figure(1)
clf
plot_them(superx,supery,w,b)
title('svm')
figure(2)
clf
plot_them(superx,supery,w2,b2)
title('reg')

function plot_them(superx,supery,w,b)
plot(superx(1,supery==1),superx(2,supery==1),'r.');
hold on;
plot(superx(1,supery==-1),superx(2,supery==-1),'g.');




mins = min(superx,[],2);
maxes = max(superx,[],2);

t = 0;
ymin = ((t + b)- w(1)*mins(1))/w(2);
ymax = ((t + b)- w(1)*maxes(1))/w(2);
plot([mins(1) maxes(1)],[ymin ymax],'k');

t = 1;
ymin = ((t + b)- w(1)*mins(1))/w(2);
ymax = ((t + b)- w(1)*maxes(1))/w(2);
plot([mins(1) maxes(1)],[ymin ymax],'r--');

t = -1;
ymin = ((t + b)- w(1)*mins(1))/w(2);
ymax = ((t + b)- w(1)*maxes(1))/w(2);
plot([mins(1) maxes(1)],[ymin ymax],'b--');

axis([mins(1) maxes(1) mins(2) maxes(2)])

function [w,b,r] = svm(X,Y)

lambda = .001;
c = 1/lambda;
supery = Y;
superx = X;
svm_model = libsvmtrain(supery,superx',sprintf('-s 0 -t 0 -c %f -q',c));
svm_weights = full(sum(svm_model.SVs .* ...
                       repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
wex = svm_weights';
w = wex;
b = svm_model.rho;
r = w'*X-b;
function [w,b,r] = reg(X,Y,K)

lambda = .00000001;
a = Y;
%A = diag(a);
saveX = X;
X(end+1,:) = 1;
curI = lambda*eye(size(X,1));
for i = 1:K
  A2 = diag(a.^2);
  %w2 = inv(X*A'*A*X' + lambda*eye(size(X,1)))*(Y'*A'*A*X')';
  
  if exist('r','var')
    oks = logical(a>0);
    w = (Y(oks)'*diag(a(oks).^2)*X(:,oks)')/(X(:,oks)*diag(a(oks).^2)*X(:,oks)' + curI);
  else
    w = (Y'*A2*X')/(X*A2*X' + curI);
  end
  w = w';

  b = -w(end);
  w = w(1:end-1);
  r = w'*saveX-b;
  
  a = Y;
  a(Y==1 & r'>1) = 0;
  a(Y==-1 & r'<-1) = 0;

  %a(Y==1) = max(1.0,a(Y==1));
  %a(Y==-1) = min(-1.0,a(Y==-1));
  %A = diag(a);
end

