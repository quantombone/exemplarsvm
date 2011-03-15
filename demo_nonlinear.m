function demo_nonlinear
%% try a nonlinear decision boundary

xs = linspace(0,4*pi,1000);
xs2 = sin(xs);

PAD = 1;
xpos = [xs; xs2 + PAD + randn(size(xs))];
xneg = [xs; xs2 - PAD + randn(size(xs))];

ypos = xs*0 + 1;
yneg = xs*0 - 1;

x = [xpos xneg];
y = [ypos yneg];

pos = find(y==1);
neg = find(y==-1);

%figure(1)
%clf
%subplot(1,2,1)

y = y';

for iii = -1:1:5
  
SVMC = 10^(-3+iii);
%SVMC = 10;
%gamma = 10^(-2+iii);
gamma = .001;



%tic
svm_model = svmtrain(y, x',sprintf(['-s 0 -t 2 -c' ...
                    ' %f -gamma %f -q'],SVMC,gamma));
%toc

%tic
[predicted_label, accuracy] = svmpredict(y, x', svm_model);
%toc



min1 = min(x,[],2);
max1 = max(x,[],2);
ranges = range(x,2);
min1 = min1 - .2*ranges;
max1 = max1 + .2*ranges;


NCUTS = 100;
[xxx,yyy] = meshgrid(linspace(min1(1),max1(1),NCUTS),...
                     linspace(min1(2),max1(2),NCUTS));

newvals = [xxx(:)'; yyy(:)'];
%tic
%[predicted_label, accuracy] = svmpredict(zeros(size(newvals,2),1), newvals', svm_model);
%toc



%tic
myscores = mysvmpredict(newvals,svm_model);
%toc
predicted_label = sign(myscores);



pos2 = find(predicted_label == 1);
neg2 = find(predicted_label == -1);

%subplot(1,2,2)
figure(1)
clf
subplot(1,2,1)
plot(newvals(1,pos2),newvals(2,pos2),'r.','MarkerSize',30)
hold on;
plot(newvals(1,neg2),newvals(2,neg2),'b.','MarkerSize',30)
hold on;
plot(x(1,pos),x(2,pos),'g.')
hold on
plot(x(1,neg),x(2,neg),'m.')
title(sprintf('Boundary SVMC=%f gamma=%f',SVMC,gamma))

subplot(1,2,2)
imagesc(reshape(myscores,NCUTS,NCUTS))
title(sprintf('#SV = %d',length(svm_model.sv_coef)))
drawnow

end


