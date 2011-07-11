function plot_voc_results(dataset_params)

m1 = [dataset_params.models_name '-svm'];
files{1} = dir([dataset_params.resdir '/' m1 '*test_results.mat']);
files{2} = dir([dataset_params.resdir '/' m1 '*calibrated' ...
                    '_test_results.mat']);

files{3} = dir([dataset_params.resdir '/' m1 '*calibrated-' ...
                    'M_test_results.mat']);

files{1} = cellfun2(@(x)sprintf('%s/%s',dataset_params.resdir,x), ...
                   {files{1}.name});
files{2} = cellfun2(@(x)sprintf('%s/%s',dataset_params.resdir,x), ...
                   {files{2}.name});
files{3} = cellfun2(@(x)sprintf('%s/%s',dataset_params.resdir,x), ...
                   {files{3}.name});

files{1} = setdiff(files{1},files{2});
files{1} = setdiff(files{1},files{3});

figure(1)
clf
for i = 1:length(files{1})
  subplot(2,10,i)
  tic
  f1 = load(files{1}{i});
  f2 = load(files{2}{i});
  f3 = load(files{3}{i});
  toc

  %figure(1)
  %clf
  plot(f1.results.recall,f1.results.prec,'k.-','LineWidth',2);
  hold on;
  plot(f2.results.recall,f2.results.prec,'g--','LineWidth',2);
  hold on;
  plot(f3.results.recall,f3.results.prec,'r-','LineWidth',2);
  grid;
  
  xlabel 'recall'
  ylabel 'precision'
  cls = dataset_params.classes{i};
  title(sprintf('class: %s, subset: %s',...
                cls,dataset_params.testset));
  
  s1 = sprintf('AP = %.3f APold=%.3f',...
               f1.results.ap,f1.results.apold);

  s2 = sprintf('AP = %.3f APold=%.3f',...
               f2.results.ap,f2.results.apold);
  
  s3 = sprintf('AP = %.3f APold=%.3f',...
               f3.results.ap,f3.results.apold);

  %title(sprintf('class: %s, subset: %s, AP = %.3f, APold=%.3f',...
  %             cls,dataset_params.testset,f.results.ap,f.results.apold));
  
  axis([0 1 0 1])
  grid on
  
  set(gca,'FontSize',16)
  set(get(gca,'Title'),'FontSize',16)
  set(get(gca,'YLabel'),'FontSize',16)
  set(get(gca,'XLabel'),'FontSize',16)
  axis([0 1 0 1.001]);
  
  legend({s1,s2,s3})
  drawnow
end

set(gcf,'PaperPosition',[0 0 40 8])
set(gcf,'PaperSize',[40 8]);
imfiler = sprintf(['%s/newall.pdf'],dataset_params.localdir);
print(gcf,imfiler,'-dpdf');
