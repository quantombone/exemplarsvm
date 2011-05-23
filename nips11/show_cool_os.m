function show_cool_os(msave)
%show the cool plots, call figure first

inds = nms_objid(msave.model.svids);

msave.model.svids = msave.model.svids(inds);
msave.model.nsv = msave.model.nsv(:,inds);


bg = get_pascal_bg('trainval');
[negatives,vals,pos,msave] = find_set_membership(msave);

%get os

fprintf(1,'getting overlaps\n');
tic
[maxos,maxind,maxclass] = get_overlaps_with_gt(msave, [msave.model.svids{:}], bg);
toc

scores = msave.model.w(:)'*msave.model.nsv-msave.model.b;
% subplot(2,1,1)
% scores = log(scores+1+eps);
% plot(pos,scores(pos),'ro')
% hold on;
% plot(vals,scores(vals),'b.')
% hold on;
% plot(negatives,scores(negatives),'k.','MarkerSize',14)

% xlabel('Rank')
% ylabel('SVM score (log(w^Tx+1)')
% title('SVM Rank versus set membership')
% legend('Positives','Validation Negatives','Hard Negatives')

%subplot(2,1,2)
scores = maxos;
plot(negatives,scores(negatives),'k.','MarkerSize',14)
hold on;
plot(vals,scores(vals),'b.')
hold on;
plot(pos,scores(pos),'ro')
xlabel('Rank')
ylabel('Maxos')
title(sprintf('Top Matching Objects for Exemplar %s.%d Object Overlap',msave.curid,msave.objectid))
drawnow
legend('Hard Negatives','Validation Negatives','Positives')
