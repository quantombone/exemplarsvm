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

scores = msave.model.w(:)'*msave.model.nsv - msave.model.b;
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

poscolor = 'g';
valcolor = 'b';
negcolor = 'r';

%subplot(2,1,2)
plot(pos,maxos(pos),[poscolor '^'],'MarkerSize',12,'MarkerFaceColor',poscolor,'MarkerEdgeColor','k')
hold on;
plot(vals,maxos(vals),[valcolor 'o'],'MarkerSize',12,'MarkerFaceColor',valcolor,'MarkerEdgeColor','k')
hold on;
plot(negatives,maxos(negatives),[negcolor 's'],'MarkerSize',10,'MarkerFaceColor',negcolor,'MarkerEdgeColor','k')

%plot os == .5 line
%plot(1:length(scores),.5,'k--')

xlabel('Rank')
ylabel('Maxos')
title(sprintf('Top Matching Objects for Exemplar %s.%d Object Overlap',msave.curid,msave.objectid))

legend('Positives','Validation Negatives','Hard Negatives')


cuts = unique(round(linspace(1,length(scores),10)));
for i = 1:length(cuts)
  labels{i} = sprintf('%.3f',scores(cuts(i)));
end
try
set(gca,'XTick',cuts);
catch
  keyboard
end
set(gca,'XTickLabel',labels)

topcuts = [0:.1:1];
for i = 1:length(topcuts)
  toplabels{i} = sprintf('%.1f',topcuts(i));
end

set(gca,'YTick',topcuts);
set(gca,'YTickLabel',toplabels)
grid on;
axis([1 length(scores) 0 1])

drawnow