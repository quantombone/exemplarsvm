
classes = {'bicycle','motorbike','horse'};
styles = {'r-','b.-','m*-'};

figure(2)
clf
for i = 1:length(classes)
  %lastval = find(cresults{i}{1}.recall>=.05);

  plot(badcresults{i}{1}.recall,badcresults{i}{1}.prc, ...
                styles{i});
  hold on;
  %plot([supportnum(i) supportnum(i)],[0 1],[styles{i} '-'],'LineWidth',13);
  %hold on;
  xlabel('Recall')
  ylabel('Precision')
  strs{i} = sprintf('via %s',classes{i});
  grid on;
  %title(strs{i})
end
title('Person Detection from non-person associations')

%title('Complementary object detection')
legend(strs)
%grid on;
