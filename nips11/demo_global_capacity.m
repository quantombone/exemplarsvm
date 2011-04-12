%% demo capacity
% Demo of Max-Margin Max-Capcity Learning
% Tomasz Malisiewicz (tomasz@cmu.edu)
function demo_global_capacity(r)

%generate synthetic data in 2D and associated classes
[x,y] = generate_synthetic_data;

%get training and testing data (Branch off the testing data and don't
%use it at train time)
[x,y,x2,y2] = split_to_train_test(x,y);

%svm_model = train_nonlinear(x,y);
svm_model = train_linear(x,y);

%generate a grid of points equal to the visible extent of the x's
[newvals,xxx] = get_grid_points(x);

%apply nonlinear svm to all grid points to produce boundary
myscores = mysvmpredict(newvals,svm_model);
myscores = reshape(myscores,size(xxx));

myscores_raw = mysvmpredict(x,svm_model);

scoremat = [];
scoremat2 = [];
ws = cell(0,1);
bs = cell(0,1);
curscores = [];

pos = find(y==1);
neg = find(y==-1);

DISPLAY = 1;
if DISPLAY == 1
  figure(1)
  clf
end
  
%% Train a max-capacity SVM for each data point
for iii = 1:length(pos)
  index = iii; 
  gamma = .1;

  xpos = x(:,y==1);
  xneg = x(:,y==-1);
    
  curx = x(:,[pos(index); neg]);
  cury = y([pos(index); neg]);
  
  SVMC = 10;
  [svm_model] = learn_local_no_capacity(curx,cury,SVMC);

  if DISPLAY == 1
    figure(1)
    clf
    subplot(1,2,1)
    plot(x(1,y==1),x(2,y==1),'r.');
    hold on;
    plot(x(1,y==-1),x(2,y==-1),'k.');
    hold on;
    plot(x(1,pos(index)),x(2,pos(index)),'kp','MarkerSize',12,'LineWidth',3)
    bbox_range = plot_decision_boundary(svm_model.w,svm_model.b,x);
    axis(bbox_range);
    drawnow
  end

  
  wold = svm_model.w;
  bold = svm_model.b;
  
  [w,b,alphas,pos_inds] = learn_local_capacity(x,y,pos(index),SVMC, ...
                                               gamma);
  
 
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
  
  curscores(end+1,:) = w'*x-b;
  scoremat(:,iii) = wold'*newvals-bold;
  scoremat2(:,iii) = w'*newvals-b;
 
   if DISPLAY == 1
    subplot(1,2,2)
    plot(x(1,y==1),x(2,y==1),'r.');
    hold on;
    plot(x(1,y==-1),x(2,y==-1),'k.');
    hold on;
    plot(x(1,pos(index)),x(2,pos(index)),'kp','MarkerSize',12,'LineWidth',3)
    bbox_range = plot_decision_boundary(w,b,x);
    if length(alphas)>0
      goods = pos_inds(alphas>0);
    else
      goods = [];
    end
   
    plot(x(1,goods),x(2,goods),'go');
    title(sprintf('M^2 Max-Capacity \\gamma=%.3f',gamma))
    axis(bbox_range);
    drawnow
    pause
  end  
end

model_linear = train_linear(x,y);

% if DISPLAY == 0
%     figure(1)
%     clf
%     plot(x(1,y==1),x(2,y==1),'r.');
%     hold on;
%     plot(x(1,y==-1),x(2,y==-1),'k.');
%     hold on;
%     bbox_range = plot_decision_boundary(model_linear.w,model_linear.b,x);
%     axis(bbox_range);
%     drawnow
% end


%x2(1,:) = (x(1,:)-min1(1)) / (max1(1)-min1(1))*NCUTS;
%x2(2,:) = (x(2,:)-min1(2)) / (max1(2)-min1(2))*NCUTS;

figure(2)
clf
subplot(2,4,1)
plot(x(1,y==1),x(2,y==1),'r.');
hold on;
plot(x(1,y==-1),x(2,y==-1),'k.');
hold on;
bbox_range = plot_decision_boundary(model_linear.w,model_linear.b,x);
axis(bbox_range);
drawnow
title('Linear SVM')

subplot(2,4,2)
imagesc(-myscores,[-1 1])
colorbar
title('Nonlinear SVM Boundary');

subplot(2,4,3)
imagesc(-reshape( max(scoremat,[],2),size(xxx)),[-1 1])
colorbar
title('ExemplarSVM Boundary')

subplot(2,4,4)
imagesc(-reshape( max(scoremat2,[],2),size(xxx)),[-1 1])
colorbar

%sigmoid = @(x)(1./(1+exp(-x)));
%imagesc(-reshape( max(sigmoid(10*scoremat2+10),[],2),size(xxx)),[-1 1])

%hold on; plot(x2(1,y==-1),x2(2,y==-1),'ko')
%hold on; plot(x2(1,y==1),x2(2,y==1),'k+')
title('M^2MC Boundary')
subplot(2,4,5)
plot(x(1,y==1),x(2,y==1),'r.');
hold on;
plot(x(1,y==-1),x(2,y==-1),'k.');
hold on;
bbox_range = plot_decision_boundary(model_linear.w,model_linear.b,x);
axis(bbox_range);
drawnow
title('Linear SVM')
subplot(2,4,6)
imagesc(-myscores)
colorbar
title('Nonlinear SVM Boundary');
subplot(2,4,7)
imagesc(-reshape( max((scoremat),[],2),size(xxx)))
colorbar
title('ExemplarSVM Boundary')
subplot(2,4,8)
imagesc(-reshape( max((scoremat2),[],2),size(xxx)))
colorbar
title('M^2MC Boundary')

% %% Compute PR curve now
% figure(3)
% clf

% rvec = cellfun2(@(f)f'*x,ws);
% rv=cat(1,rvec{:});
% [aa,bb] = sort(max(rv,[],1),'descend');
% cury = y;
% corr = cury(bb);
% plot(cumsum(corr)./(1:length(corr))')

% myscores_raw = mysvmpredict(x,svm_model);

% [aa,bb] = sort(myscores_raw,'descend');
% corr = cury(bb);
% hold all;
% plot(cumsum(corr)./(1:length(corr))')
% legend('Max Capacity','Kernel SVM')

% %%
  
function [svm_model] = learn_local_no_capacity(x,y,SVMC)
% Perform monolithic training


svm_model = libsvmtrain(y, x',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'],SVMC));

svm_weights = full(sum(svm_model.SVs .* ...
                       repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
w = svm_weights';
b = svm_model.rho;
if y(1) == -1
  w = w*-1;
  y = y*-1;
end
svm_model.w = w;
svm_model.b = b;

function x=boost(x)
%% Explicit second order feature boosting
%
x = x;
return;
x(3,:) = x(1,:).*x(2,:);
x(4,:) = x(1,:).*x(1,:);
x(5,:) = x(2,:).*x(2,:);


function [x,y] = generate_synthetic_data
%% Generate a 2D synthetic point set
% Generate synthetic data
t = linspace(0,2*pi,100);
xpos = [12*cos(t); 12*sin(t)];
xneg = [3*cos(t); 22*sin(t)]; 
if exist('r','var')
  [u,v] = find(r==1);
  xneg = [u(:)'; v(:)'];
  
  [u,v] = find(r==0);
  xpos = [u(:)'; v(:)'];
end

rand('seed',1234);
xpos = xpos + 1.5*randn(size(xpos));
xneg = xneg + 0.5*randn(size(xneg));

ypos = ones(size(xpos,2),1)'*0+1;
yneg = ones(size(xneg,2),1)'*0-1;
x = [xpos xneg];
y = [ypos yneg];
y = y';
%y = y*-1;
%boost features to higher dimensions explicitly
x2 = x;
y2 = y;
x2 = x2 + 50;
x = cat(2,x,x2);
y = cat(1,y,y2);

x = boost(x);

function [x,y,x2,y2] = split_to_train_test(x,y)
r = randperm(length(y));
rneg = r(1:(length(r)/2));
x2 = x(:,rneg);
y2 = y(rneg);
x(:,rneg) = [];
y(rneg) = [];

function svm_model = train_nonlinear(x,y)
%% Train non-linear SVM
SVM_GAMMA = .001;
SVMC = .01;

r = find(randn(length(y),1)>0);
x2 = x(:,r);
y2 = y(r);
x(:,r) = [];
y(r) = [];

p1 = [100 10 1 .1 .01];
p2 = [100 10 1 .1 .01];
bestscore = -1;
bestmodel = [];
for i = 1:length(p1)
  for j = 1:length(p2)
    SVMC = p1(i);
    SVM_GAMMA = p2(j);
    svm_model = libsvmtrain(y, x',sprintf(['-s 0 -t 2 -c' ...
                    ' %f -gamma %f -q'],SVMC,SVM_GAMMA));
    
    res2 = mysvmpredict(x2,svm_model);
    [aa,bb] = sort(res2,'descend');
    corr = (y2(bb)==1);
    ap = (cumsum(corr)./(1:length(corr))');
    map = mean(ap);
    
    %keyboard
    %score = mean(sign(res2(:))==y2(:));
    score = map;
    %fprintf(1,'score is %.4f\n',score);
    if score > bestscore
      bestscore = score;
      bestmodel = svm_model;
    end

  end
end
svm_model = bestmodel;
fprintf(1,'best score is %.4f\n',bestscore);

function svm_model = train_linear(x,y)
%% Train non-linear SVM
r = find(randn(length(y),1)>0);
x2 = x(:,r);
y2 = y(r);
x(:,r) = [];
y(r) = [];

p1 = [100 10 1 .1 .01];

bestscore = -1;
bestmodel = [];
for i = 1:length(p1)

    SVMC = p1(i);

    svm_model = learn_local_no_capacity(x,y,SVMC);
    
    %res2 = mysvmpredict(x2,svm_model);
    
    res2 = mysvmpredict(x2,svm_model);
    [aa,bb] = sort(res2,'descend');
    corr = (y2(bb)==1);
    ap = (cumsum(corr)./(1:length(corr))');
    map = mean(ap);


    %score = mean(sign(res2(:))==y2(:));
    score = map;
    %fprintf(1,'score is %.4f\n',score);
    if score > bestscore
      bestscore = score;
      bestmodel = svm_model;
    end
end
svm_model = bestmodel;

function [newvals,xxx] = get_grid_points(x)
min1 = min(x,[],2);
max1 = max(x,[],2);
ranges = range(x,2);
min1 = min1 - .2*ranges;
max1 = max1 + .2*ranges;

%Generate a NCUTS x NCUTS grid
NCUTS = 100;
[xxx,yyy] = meshgrid(linspace(min1(1),max1(1),NCUTS),...
                     linspace(min1(2),max1(2),NCUTS));
newvals = [xxx(:)'; yyy(:)'];
newvals(2,:) = newvals(2,end:-1:1);

newvals = boost(newvals);

function bbox_range = plot_decision_boundary(w,b,x)
  
xmin = min(x(1,:));
xmax = max(x(1,:));
PAD = .4*(max(x(1,:))-min(x(1,:)));
bbox_range = [min(x(1,:))-PAD max(x(1,:))+PAD min(x(2,:))-PAD max(x(2, ...
                                                  :))+PAD];

xmin = bbox_range(1);
xmax = bbox_range(2);

offsets = [0 -1 1];
styles = {'k','r--','b--'};

for q = 1:length(offsets)

ymin = (offsets(q)+b - w(1)*xmin)/w(2);
ymax = (offsets(q)+b - w(1)*xmax)/w(2);

%yminneg = (1+b - w(1)*bbox_range(1))/w(2);
%ymaxneg = (1+b - w(1)*bbox_range(2))/w(2);

%yminpos = (-1+b - w(1)*bbox_range(1))/w(2);
%ymaxpos = (-1+b - w(1)*bbox_range(2))/w(2);

hold on;
plot([xmin xmax],[ymin ymax],styles{q},'LineWidth',2)
%hold on;
%plot([xmin xmax],[yminneg ymaxneg],'b--','LineWidth',2)
%hold on;
%plot([xmin xmax],[yminpos ymaxpos],'r--','LineWidth',2)
end