function demo_capacity(r)
%% try a nonlinear decision boundary

SVMC = 1;

%SVMC = 10;
%xs = linspace(0,4*pi,1000);
%xs2 = sin(xs);
rand('seed',1234);
t = linspace(0,2*pi,100);
%xpos = [t; 10*sin(t)];
%xneg = [cos(t); 10*sin(t)-5];
xpos = [12*cos(t); 12*sin(t)];
xneg = [3*cos(t); 13*sin(t)]; 

if exist('r','var')
  [u,v] = find(r==1);
  xneg = [u(:)'; v(:)'];
  
  [u,v] = find(r==0);
  xpos = [u(:)'; v(:)'];
end

%xpos2 = [1+8*cos(t); 2*sin(t)];
xpos2 = [];

xpos = [xpos xpos2];

xpos = xpos + 0.5*randn(size(xpos));
xneg = xneg + 0.5*randn(size(xneg));

%xpos = xpos*10;
%xneg = xneg*10;
%PAD = 1;
%xpos = [xs; xs2 + PAD + randn(size(xs))];
%xneg = [xs; xs2 - PAD + randn(size(xs))];

ypos = ones(size(xpos,2),1)'*0+1;
yneg = ones(size(xneg,2),1)'*0-1;

x = [xpos xneg];
y = [ypos yneg];


x = boost(x);
pos = find(y==1);
neg = find(y==-1);

%figure(1)
%clf
%subplot(1,2,1)

curscores = [];
y = y';

%%%%%%%%
gamma = .1;
halfers = randn(size(y))>0;
svm_model = libsvmtrain(y(halfers), x(:,halfers)',sprintf(['-s 0 -t 2 -c' ...
                    ' %f -gamma %f -q'],SVMC,gamma));
%toc

%tic
%[predicted_label, accuracy] = svmpredict(y, x', svm_model);
%toc


min1 = min(x,[],2);
max1 = max(x,[],2);
ranges = range(x,2);
min1 = min1 - .2*ranges;
max1 = max1 + .2*ranges;

NCUTS = 100;
[xxx,yyy] = meshgrid(linspace(min1(1),max1(1),NCUTS),...
                     linspace(min1(2),max1(2),NCUTS));

%newvals = [xxx(:)'; yyy(:)'];
newvals = [yyy(:)'; xxx(:)'];
newvals(1,:) = newvals(1,end:-1:1);
newvals = boost(newvals);

%tic
%[predicted_label, accuracy] = svmpredict(zeros(size(newvals,2),1), newvals', svm_model);
%toc



%tic


myscores = mysvmpredict(boost(newvals),svm_model);
%toc
predicted_label = sign(myscores);
myscores = reshape(myscores,size(xxx));

myscores_raw = mysvmpredict(x,svm_model);



%%%%%%%


scoremat = [];
scoremat2 = [];
ws = cell(0,1);
bs = cell(0,1);
for iii = 1:size(xpos,2)%1:10000
  index = iii; %s1+mod(iii,length(t));
  gamma = 1;
  %gamma = gammas(iii); %.001;
  %gammaK = iii;
  %gammaK = 0;


  
xpos = x(:,y==1);
xneg = x(:,y==-1);

figure(1)
clf
subplot(1,3,1)
imagesc(-myscores)

subplot(1,3,2)
plot(x(1,y==-1),x(2,y==-1),'r.')
hold on;
plot(x(1,y==1),x(2,y==1),'b+')

hold on;
plot(x(1,index),x(2,index),'kp','MarkerSize',12,'LineWidth',3)


ds = distSqr_fast(xpos(:,index),xpos);
[aa,bb] = sort(ds);

K = 1;
curx = [xpos(:,bb(1:K)) xneg];
cury = [ypos(bb(1:K)) yneg]';

[w,b] = learn_local_no_capacity(curx,cury,index,SVMC);


res = w'*x-b;

xmin = min(x(1,:));
xmax = max(x(1,:));


PAD = .1*(max(x(1,:))-min(x(1,:)));
bbox_range = [min(x(1,:))-PAD max(x(1,:))+PAD min(x(2,:))-PAD max(x(2, ...
                                                  :))+PAD];

xmin = bbox_range(1);
xmax = bbox_range(2);
ymin = (b - w(1)*xmin)/w(2);
ymax = (b - w(1)*xmax)/w(2);

% bbox_range(1:3) = bbox_range(1:3)-10;
% bbox_range(2:4) = bbox_range(2:4)+10;


yminneg = (1+b - w(1)*bbox_range(1))/w(2);
ymaxneg = (1+b - w(1)*bbox_range(2))/w(2);

yminpos = (-1+b - w(1)*bbox_range(1))/w(2);
ymaxpos = (-1+b - w(1)*bbox_range(2))/w(2);

hold on;
plot([xmin xmax],[ymin ymax],'k','LineWidth',2)
hold on;
plot([xmin xmax],[yminneg ymaxneg],'b--','LineWidth',2)
hold on;
plot([xmin xmax],[yminpos ymaxpos],'r--','LineWidth',2)
title(sprintf('ExemplarSVM Max-Margin Solution'))
axis(bbox_range)
subplot(1,3,3)
plot(x(1,y==-1),x(2,y==-1),'r.')
hold on;
plot(x(1,y==1),x(2,y==1),'b+')

hold on;
plot(x(1,index),x(2,index),'kp','MarkerSize',12,'LineWidth',3)

wold = w;
bold = b;
[w,b,alphas,pos_inds] = learn_local_capacity(x,y,index,SVMC,gamma);

ws{end+1} = w;
bs{end+1} = b;

alphas = logical(alphas);
%[w,b,alphas,pos_inds] = learn_local_rank_capacity(x,y,index,SVMC, ...
%                                                  gamma);


%r = w'*x(:,pos_inds)-b;
%[aa,bb] = sort(r,'descend');
%alphas = bb(1:10);

alphas2 = zeros(size(pos_inds));
alphas2(alphas) = 1;
alphas = alphas2;
sum(alphas(pos_inds))
length(pos_inds)




curscores(end+1,:) = w'*x-b;


%newvals = [xxx(:)'; yyy(:)'];
scoremat(:,iii) = wold'*newvals-bold;
scoremat2(:,iii) = w'*newvals-b;

res = w'*x-b;

xmin = bbox_range(1);
xmax = bbox_range(2);
ymin = (b - w(1)*xmin)/w(2);
ymax = (b - w(1)*xmax)/w(2);

yminneg = (1+b - w(1)*xmin)/w(2);
ymaxneg = (1+b - w(1)*xmax)/w(2);

yminpos = (-1+b - w(1)*xmin)/w(2);
ymaxpos = (-1+b - w(1)*xmax)/w(2);

yminneg = (1+b - w(1)*bbox_range(1))/w(2);
ymaxneg = (1+b - w(1)*bbox_range(2))/w(2);

yminpos = (-1+b - w(1)*bbox_range(1))/w(2);
ymaxpos = (-1+b - w(1)*bbox_range(2))/w(2);


hold on;
plot([xmin xmax],[ymin ymax],'k','LineWidth',2)
hold on;
plot([xmin xmax],[yminneg ymaxneg],'b--','LineWidth',2)
hold on;
plot([xmin xmax],[yminpos ymaxpos],'r--','LineWidth',2)
hold on;

if length(alphas)>0
  goods = pos_inds(alphas>0);
else
  goods = [];
end
plot(x(1,goods),x(2,goods),'go');
title(sprintf('M^2 Max-Capacity \\gamma=%.3f',gamma))
axis(bbox_range);

index
drawnow
%pause

end

 x2(1,:) = (x(1,:)-min1(1)) / (max1(1)-min1(1))*NCUTS;
 x2(2,:) = (x(2,:)-min1(2)) / (max1(2)-min1(2))*NCUTS;

 figure(2)
 subplot(2,3,1)
 imagesc(-myscores,[-1 1])
 hold on; plot(x2(1,y==-1),x2(2,y==-1),'ko')
 hold on; plot(x2(1,y==1),x2(2,y==1),'k+')
 title('Nonlinear SVM Boundary');
 subplot(2,3,2)
 imagesc(-reshape( max(scoremat,[],2),size(xxx)),[-1 1])
 hold on; plot(x2(1,y==-1),x2(2,y==-1),'ko')
 hold on; plot(x2(1,y==1),x2(2,y==1),'k+')
 title('ExemplarSVM Boundary')
 subplot(2,3,3)
 imagesc(-reshape( max(scoremat2,[],2),size(xxx)),[-1 1])
 hold on; plot(x2(1,y==-1),x2(2,y==-1),'ko')
 hold on; plot(x2(1,y==1),x2(2,y==1),'k+')
 title('M^2MC Boundary')
 subplot(2,3,4)
 imagesc(-myscores)
 title('Nonlinear SVM Boundary');
 subplot(2,3,5)
 imagesc(-reshape( max(scoremat,[],2),size(xxx)))
 title('ExemplarSVM Boundary')
 subplot(2,3,6)
 imagesc(-reshape( max(scoremat2,[],2),size(xxx)))
 title('M^2MC Boundary')
 
 %% show pr curve now
 figure(3)
 clf
 
 rvec = cellfun2(@(f)f'*x(:,~halfers),ws);
 rv=cat(1,rvec{:});
 [aa,bb] = sort(max(rv,[],1),'descend');
 cury = y(~halfers);
 corr = cury(bb);
 plot(cumsum(corr)./(1:length(corr))')
 
 myscores_raw = mysvmpredict(boost(x(:,~halfers)),svm_model);
 
 [aa,bb] = sort(myscores_raw,'descend');
 corr = cury(bb);
 hold all;
 plot(cumsum(corr)./(1:length(corr))')
 legend('Max Capacity','Kernel SVM')
 

 

 
function [w,b] = learn_local_no_capacity(x,y,index,SVMC)

pos_inds = find(y==1);
alphas = y*0+1;

svm_model = libsvmtrain(y, x',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'],SVMC));

svm_weights = full(sum(svm_model.SVs .* ...
                       repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
w = svm_weights';
b = svm_model.rho;

function x=boost(x)
x = x;
%x(3,:) = x(1,:).*x(2,:);
%x(4,:) = x(1,:).*x(1,:);
%x(5,:) = x(2,:).*x(2,:);
