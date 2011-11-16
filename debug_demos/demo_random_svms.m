function demo_random_svms
%% Here is a simple test where we generate N datapoints in a space
%% of D dimensions, and we try to train a classifier using a linear
%% SVM when assigning classes RANDOMLY to the datapoints

%% The take-home lesson is that if N<<D, then we can always create
%% a perfect separation between the positives and negatives

%% Tomasz Malisiewicz

D = 100;
N = 100;
superx = randn(D,N);
supery = double(rand(N,1)>.9);
supery(1) = 1;
supery(supery==0) = -1;
[supery,ind] = sort(supery,'descend');
superx = superx(:,ind);

if 0
svm_model = svmtrain(supery,superx','-s 0 -t 0 -c .01 -q');
svm_weights = full(sum(svm_model.SVs .* ...
                       repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
wex = svm_weights';
b = svm_model.rho;
end

%% try liblinear now
%model2 = liblinear_train(supery, sparse(superx)', ['-s 2 -B 1 -c' ...
%                    ' .1'])

[w,b,obj] = primal_svm_pos(supery,1/.1,superx');
b = -b;

model = liblinear_train(supery, sparse(superx)', '-s 2 -B 1 -c .1')
w2 = model.w(1:end-1)';
b2 = model.w(end);

figure(4)
plot(w,w2,'r.')

y2 = w'*superx - b;
figure(1)
clf
pos = find(supery==1);
neg = find(supery==-1);
plot(1+randn(length(pos))*.01,y2(pos),'r.');
hold on;
plot(-1+randn(length(neg))*.01,y2(neg),'g.');
xlabel('Class')
ylabel('Returned SVM score')
title(sprintf('Random SVM: N=%d, D=%d',N,D))