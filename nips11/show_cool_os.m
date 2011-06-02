function show_cool_os(msave)
%show the cool os score plot, call figure first

figure(40)
clf
minestring = '';
if isfield(msave,'mining_stats')
  TOTAL = 0;
  for i = 1:length(msave.mining_stats)
    TOTAL = TOTAL + msave.mining_stats{i}.num_violating;
    TOTAL = TOTAL + msave.mining_stats{i}.num_empty;
  end
  minestring = sprintf('#Mined=%d ',TOTAL);
end

msave.model.target_id{1}.set = 3;
msave.model.target_id{1}.maxos = 1;
msave.model.target_id{1}.maxind = -1;
msave.model.target_id{1}.maxclass = -1;

%% add self
%msave.model.svids = cat(2,msave.model.target_id(1),...
%                        msave.model.svids);
%msave.model.nsv = cat(2,msave.model.target_x(:,1),...
%                      msave.model.nsv);

inds = nms_objid(msave.model.svids);

msave.model.svids = msave.model.svids(inds);
msave.model.nsv = msave.model.nsv(:,inds);


%if isfield(model.svids,'set')
sets = cellfun(@(x)x.set,msave.model.svids);
%else
%  sets = cellfun(@(x)1,msave.model.svids);
%end
negatives = find(sets==1);
vals = find(sets==2);
pos = find(sets==3);
%[negatives,vals,pos] = find_set_membership(msave.model.svids,msave.cls);

maxos = cellfun(@(x)x.maxos,msave.model.svids);
maxind = cellfun(@(x)x.maxind,msave.model.svids);
maxclass = cellfun(@(x)x.maxclass,msave.model.svids);

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


% plot(pos,maxos(pos),[poscolor '^'],'MarkerSize',12,'MarkerFaceColor',poscolor,'MarkerEdgeColor','k')
% hold on;
% plot(vals,maxos(vals),[valcolor 'o'],'MarkerSize',12,'MarkerFaceColor',valcolor,'MarkerEdgeColor','k')
% hold on;
% plot(negatives,maxos(negatives),[negcolor 's'],'MarkerSize',10, ...
%      'MarkerFaceColor',negcolor,'MarkerEdgeColor','k')

plot(negatives,maxos(negatives),[negcolor 'x'],'LineWidth',5, ...
     'MarkerSize',18)
hold on;
plot(vals,maxos(vals),[valcolor '^'],'MarkerSize',12, ...
     'MarkerFaceColor',valcolor)
hold on;
plot(pos,maxos(pos),[poscolor '+'],'LineWidth',5,'MarkerSize',18)

set(gca,'FontSize',16)
xlabel('{\bf w}^T{\bf x}+b','FontSize',18)
ylabel('GT Overlap','FontSize',18)
title(sprintf('%sDetection OS vs. Score for %s.%d',...
              minestring,msave.curid,msave.objectid),'FontSize',18)
%legend('Positives','Validation Negatives','Hard Negatives')

cuts = unique(round(linspace(1,length(scores),10)));
for i = 1:length(cuts)
  labels{i} = sprintf('%.3f',scores(cuts(i)));
end


set(gca,'XTick',cuts);
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