function [status,m,Isv] = pool_train(fff)
%use the worker pool to do super fast training

status = 0;
%BASEDIR = '/nfs/baikal/tmalisie/pool/';
BASEDIR = get_pool_directory;
%filer = dir([BASEDIR fff]);
file = [BASEDIR fff];
m = load(file);
m = m.m;

mold = m;
subs = dir([BASEDIR '/pool/*' fff]);
length(subs)
resser = cell(length(subs),1);
for i = 1:length(subs)
  fprintf(1,'.');
  resser{i} = load([BASEDIR '/pool/' subs(i).name]);
end

mining_params = get_default_mining_params;
mining_params.BALANCE_POSITIVES = 0;
mining_params.SVMC = .01;
mining_params.SVMC = 1;
mining_params.extract_negatives = 1;

%%NOTE hardcoded to use one exemplar, not a stack of multiple
xs = cellfun2(@(x)x.hn.xs{1},resser);
objids = cellfun2(@(x)x.hn.objids{1},resser);

xs = cat(2,xs{:});
objids = cat(2,objids{:});
if length(objids) == 0
  fprintf(1,'warning empty objids\n');
end

m = add_new_detections(m,xs,objids);
NNN = 12;
bg = get_pascal_bg('trainval');
%Isv = get_sv_stack(m,bg,NNN);

msave = m;
[m] = do_svm(m, mining_params);

[aa2,bb2] = sort(msave.model.w(:)'*m.model.nsv-msave.model.b,'descend');
[aa3,bb3] = sort(m.model.w(:)'*m.model.nsv-msave.model.b,'descend');
bb2 = bb2(1:100);
bb3 = bb3(1:100);
fprintf(1,'overlap is %.3f \n', length(intersect(bb2,bb3))/length(bb2));
if length(intersect(bb2,bb3))/length(bb2)>.8
  status = 1;
  %% we are done training if the sets are the same
end
  
 
%m.model.w = reshape(wex,size(m.model.w));
%m.model.b = b;
Isv = get_sv_stack(m,bg,NNN);
figure(1)
clf
imagesc(Isv);
drawnow

% r = m.model.w(:)'*superx-m.model.b;
% r2 = mold.model.w(:)'*superx-mold.model.b;
% xxx = 1:length(rrr);
% figure(2)
% clf
% plot(xxx,r2,'b.')
% hold on;
% drawnow

r = m.model.w(:)'*msave.model.nsv - m.model.b;

N = sum(r>=-1);
[scores,bb] = sort(r,'descend');

scores = scores(1:N);
r = scores;

msave.model.nsv = msave.model.nsv(:,bb(1:N));
msave.model.svids = msave.model.svids(bb(1:N));

[negatives,vals,pos] = find_set_membership(msave);

%get os

fprintf(1,'getting overlaps\n');
tic
[maxos,maxind,maxclass] = get_overlaps_with_gt(msave, [msave.model.svids{:}], bg);
toc

figure(2)
clf
subplot(2,1,1)
scores = log(scores+1+eps);
plot(pos,scores(pos),'ro')
hold on;
plot(vals,scores(vals),'b.')
hold on;
plot(negatives,scores(negatives),'k.','MarkerSize',14)

xlabel('Rank')
ylabel('SVM score (log(w^Tx+1)')
title('SVM Rank versus set membership')
legend('Positives','Validation Negatives','Hard Negatives')

subplot(2,1,2)
scores = maxos;
plot(negatives,scores(negatives),'k.','MarkerSize',14)
hold on;
plot(vals,scores(vals),'b.')
hold on;
plot(pos,scores(pos),'ro')
xlabel('Rank')
ylabel('Maxos')
title('Object Overlap')
drawnow

if ~isfield(m.model,'K')
  m.model.K = 10;
end

%VOCinit;
%target = find(ismember(m.cls,VOCopts.classes));
%[aa,bb] = sort((maxos>.5)+r'+(maxclass==target),'descend');
%% find top K things, but do nms on them
%write function that does NMS on a set of detections from multiple images

%m.model.x = msave.model.nsv(:,bb(1:m.model.K));

[m] = do_svm(m, mining_params);
save(file,'m');
