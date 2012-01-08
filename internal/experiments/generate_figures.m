function [dalals,results]=generate_figures(dalals,results)
basedir = '/nfs/baikal/tmalisie/results/VOC2007/';

if ~exist('results','var')
load([basedir '/dalal_results.mat']);
dalals = results;

VOCinit;

files = dir([basedir '*iccv*mat']);

figure(1)
clf

for i = 1:length(files)

  fprintf(1,'.');
  r=load([basedir files(i).name]);
  results{i} = r.results;
  subplot(2,10,i)
  plot(results{i}.recall,results{i}.prec,'r','LineWidth',18)
  hold on;
  plot(dalals{i}.recall,dalals{i}.prec,'k--','LineWidth',18)
  axis([0 1 0 1])
  grid;
  title(VOCopts.classes{i})
  drawnow

end

end
VOCinit
for i = 1:length(results)

  fprintf(1,'.');
  subplot(2,10,i)
  plot(results{i}.recall,results{i}.prec,'r');
  hold on;
  plot(dalals{i}.recall,dalals{i}.prec,'b--');
  axis([0 1 0 1])
  grid on;
  legend(sprintf(['E: AP=' ...
                 ' %.3f'],results{i}.apold),...
         sprintf(['DT: AP=' ...
                  ' %.3f'],dalals{i}.apold));

  xlabel 'recall'
  ylabel 'precision'
  title(VOCopts.classes{i})
  drawnow

end


set(gcf,'PaperPosition',[0 0 40 8])
set(gcf,'PaperSize',[40 8]);
imfiler = sprintf(['/nfs/baikal/tmalisie/newall.pdf']);
print(gcf,imfiler,'-dpdf');
