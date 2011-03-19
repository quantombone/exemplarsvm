function demo_capacity
%% try a nonlinear decision boundary

SVMC = .1;
%xs = linspace(0,4*pi,1000);
%xs2 = sin(xs);
t = linspace(0,2*pi,300);
xpos = [10*cos(t); 10*sin(t)]
xneg = [3*cos(t); 3*sin(t)];

xpos = xpos + randn(size(xpos));
xneg = xneg + randn(size(xneg));

%PAD = 1;
%xpos = [xs; xs2 + PAD + randn(size(xs))];
%xneg = [xs; xs2 - PAD + randn(size(xs))];

ypos = t*0 + 1;
yneg = t*0 - 1;

x = [xpos xneg];
y = [ypos yneg];

pos = find(y==1);
neg = find(y==-1);

%figure(1)
%clf
%subplot(1,2,1)

y = y';

gammas = linspace(.00001,100,size(x,2));
for iii = 1:1:size(xpos,2)
  index = iii
  gamma = 1;
  %gamma = gammas(iii); %.001;
  %gammaK = iii;
  %gammaK = 0;


figure(1)
clf
subplot(1,2,1)
plot(x(1,y==-1),x(2,y==-1),'r.')
hold on;
plot(x(1,y==1),x(2,y==1),'g.')

hold on;
plot(x(1,index),x(2,index),'kp','MarkerSize',12,'LineWidth',3)

ds = distSqr_fast(xpos(:,index),xpos);
[aa,bb] = sort(ds);
curx = [xpos(:,bb(1:50)) xneg];
cury = [ypos(bb(1:50)) yneg]';

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
title('Max-Margin Solution')
axis(bbox_range)
subplot(1,2,2)
plot(x(1,y==-1),x(2,y==-1),'r.')
hold on;
plot(x(1,y==1),x(2,y==1),'g.')

hold on;
plot(x(1,index),x(2,index),'kp','MarkerSize',12,'LineWidth',3)

[w,b,alphas,pos_inds] = learn_local_capacity(x,y,index,SVMC,gamma);

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


goods = pos_inds(alphas>0);

plot(x(1,goods),x(2,goods),'ko');
title(sprintf('Max-Capacity \\gamma=%.3f',gamma))

axis(bbox_range)

drawnow

end


function [w,b,alphas,pos_inds] = learn_local_capacity(x,y,index,SVMC,gamma)
%maximum-capacity learning
pos_inds = find(y==y(index));

flip = 0;
if y(index) == -1
  flip = 1;
  y = y *-1;
end

g = ones(size(pos_inds));

%turn all negatives on, they always stay on
%turn all positives off at start
alphas = y*0+1;
alphas(pos_inds)=0;

%turn self on (its a positive)
alphas(index) = 1;

oldgoods = [];
for k = 1:20
  goods = find(alphas==1);
  if length(oldgoods) > 0
    diffinds = setdiff(goods,oldgoods);
    if length(diffinds) == 0
      alphas = alphas(pos_inds);
      return;
    end
  end

  oldgoods = goods;
  frac=(sum(alphas)-sum(y==-1)) / length(pos_inds);
  
  svm_model = svmtrain(y(goods), x(:,goods)',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'],SVMC));
  
  
  svm_weights = full(sum(svm_model.SVs .* ...
                         repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
  w = svm_weights';
  b = svm_model.rho;
  
  if 1
    %%optimize alphas 
    if 1
      %gamma term
      hinge = @(x)max(1-x,0.0);
      loss_term = hinge((w'*x(:,pos_inds)) .* y(pos_inds)');
      alphas(pos_inds) = loss_term < gamma*g';  
    % else
    %   %%TOPK
    %   loss_term = (1-(w'*x(:,pos_inds)) .* y(pos_inds)');
    %   [alpha,beta] = sort(loss_term);
    %   alphas(pos_inds) = 1;
    %   alphas(pos_inds(beta(K+1:end)))=0;
    end
    
    %fprintf(1,'hack non-enable sellf\n');
    %turn self on
    %alphas(index) = 1;
  end
  
  %fprintf(1,'frac +: %.3f, raw=%d\n',frac,sum(alphas)-sum(y==-1));
  
end

fprintf(1,' --frac +: %.3f, raw=%d\n',frac,sum(alphas)-sum(y==-1));

if flip == 1
  w = w*-1;
  b = b*-1;
end

alphas = alphas(pos_inds);

function [w,b] = learn_local_no_capacity(x,y,index,SVMC)

pos_inds = find(y==1);
alphas = y*0+1;

svm_model = svmtrain(y, x',sprintf(['-s 0 -t 0 -c' ...
                    ' %f -q'],SVMC));

svm_weights = full(sum(svm_model.SVs .* ...
                       repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
w = svm_weights';
b = svm_model.rho;

