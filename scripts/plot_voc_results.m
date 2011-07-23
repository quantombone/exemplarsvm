function plot_voc_results(dataset_params)

c = dataset_params.classes;

names{5} = [dataset_params.models_name '-svm.%s_test_results.mat'];
names{6} = [dataset_params.models_name '-svm.%s-calibrated_test_results.mat'];
names{7} = [dataset_params.models_name '-svm.%s-calibrated-M_test_results.mat'];

names{1} = [dataset_params.models_name '-normalizedhog.%s_test_results.mat'];
names{2} = [dataset_params.models_name '-normalizedhog.%s-calibrated_test_results.mat'];
%names{6} = [dataset_params.models_name '-normalizedhog.%s-calibrated-M_test_results.mat'];

names{3} = [dataset_params.models_name '-dfun.%s_test_results.mat'];
names{4} = [dataset_params.models_name '-dfun.%s-calibrated_test_results.mat'];
%names{9} = [dataset_params.models_name '-dfun.%s-calibrated-M_test_results.mat'];

%c = c(1:3);
for i = 1:length(c)
  for j = 1:length(names)
    f = dir([dataset_params.resdir '/' sprintf(names{j},c{i})]);
    if length(f) == 0
      aps(i,j) = -.01;
    else
      f = load([dataset_params.resdir '/' f(1).name]);
      aps(i,j) = f.results.apold;
    end
  end
  aps(i,:);
  fprintf(1,'.');
end

dt = [.127 .253 .005, .015, .107 .205 .23 .005 .021 .128 .014 .004 ...
      .122 .103 .101 .022 .056 .05 .12 .248];
ldpm = [.287 .51 .006 .145 .265 .397 .502 .163 .165 .166 .245 .05 ...
        .452 .383 .362 .09 .174 .228 .341 .384]

aps(:,end+1) = dt(1:length(c));
aps(:,end+1) = ldpm(1:length(c));

imfiler = sprintf(['%s/res_table2.tex'],dataset_params.localdir);
fid = fopen(imfiler,'w');
fprintf(fid,['\\footnotesize\n\\begin{table}\n\\begin{center}\n\\begin{tabular}{|l||' ...
             'c|c|c|c|c|c|c|c|c|}\n\\hline\n']);
fprintf(fid,'& \\multicolumn{7}{|c|}{Per-Exemplar Methods} & \\multicolumn{2}{|c|}{Global}\\\\\n');
fprintf(fid,'\\hline\n');
fprintf(fid,'Type & NNHOG & NNHOG+B & DFUN & DFUN+B & ESVM & ESVM+B & ESVM+M & DT & LDPM\\\\\n');
fprintf(fid,'\\hline\n');

aps2 = aps;
aps2(aps2<0) = nan;
aps(aps<0) = -.00000001;

means = nanmean(aps,1);
%c = dataset_params.classes;
for i = 1:length(c)
  fprintf(fid,'%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f\\\\\n',...
          c{i},aps(i,1),aps(i,2),aps(i,3),aps(i,4),aps(i,5),aps(i,6),aps(i,7),aps(i,8),aps(i,9));
end
fprintf(fid,'\\hline\n');
fprintf(fid,'mAP & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f\\\\\n',...
        means(1), means(2), means(3), means(4), means(5), means(6), ...
        means(7), means(8), means(9));
        
        % mean(aps(:,1)),mean(aps(:,2)),mean(aps(:,3)),mean(aps(:, ...
        %                                           4)),mean(aps(:,5)),mean(aps(:,6)),...
        % mean(aps(:,7)),mean(aps(:,8)),mean(aps(:,9)));
fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n\\end{center}\n\\caption{hi}\n\\end{table}');
fclose(fid);

fprintf(1,'keyboard in plot results\n');
keyboard

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
ha = tight_subplot(5, 4, .05, ...
                   .1, .01);

aps = [];
for i = 1:length(files{1})
  %subplot(2,10,i)

  axes(ha(i));
  tic
  f1 = load(files{1}{i});
  f2 = load(files{2}{i});
  f3 = load(files{3}{i});
  toc

  %figure(1)
  %clf
  myplot(f1.results.recall,f1.results.prec,'k.-','LineWidth',2);
  hold on;
  myplot(f2.results.recall,f2.results.prec,'g--','LineWidth',2);
  hold on;
  myplot(f3.results.recall,f3.results.prec,'r-','LineWidth',2);
  grid;
  

  xlabel 'recall'
  ylabel 'precision'
  cls = dataset_params.classes{i};
  %title(sprintf('class: %s, subset: %s',...
  %              cls,dataset_params.testset));
  
  title(sprintf('%s',...
                cls))

  s1 = sprintf('ESVM: AP=%.3f',...
               f1.results.apold);

  s2 = sprintf('ESVM+B: AP=%.3f',...
               f2.results.apold);
  
  s3 = sprintf('ESVM+B+M:+AP=%.3f',...
               f3.results.apold);

  aps(i,1) = f1.results.apold;
  aps(i,2) = f2.results.apold;
  aps(i,3) = f3.results.apold;
  
  % s1 = sprintf('AP = %.3f APold=%.3f',...
  %              f1.results.ap,f1.results.apold);

  % s2 = sprintf('AP = %.3f APold=%.3f',...
  %              f2.results.ap,f2.results.apold);
  
  % s3 = sprintf('AP = %.3f APold=%.3f',...
  %              f3.results.ap,f3.results.apold);

  %title(sprintf('class: %s, subset: %s, AP = %.3f, APold=%.3f',...
  %             cls,dataset_params.testset,f.results.ap,f.results.apold));
  

  grid on
  
  set(gca,'FontSize',16)
  set(get(gca,'Title'),'FontSize',16)
  set(get(gca,'YLabel'),'FontSize',16)
  set(get(gca,'XLabel'),'FontSize',16)
  axis([0 1 0 1.001]);
  
  legend({s1,s2,s3})
  drawnow
end

set(gcf,'PaperPosition',[0 0 20 40])
set(gcf,'PaperSize',[20 40]);
imfiler = sprintf(['%s/newall.pdf'],dataset_params.localdir);
print(gcf,imfiler,'-dpdf');

figure(2)
clf
bar(aps)
axis([0 21 0 .5])
set(gca,'XTick',1:21);
set(gca,'XTickLabel','');

blanky = '';
for q = 1:20
  blanks{q}=blanky;
end

set(gca,'XTickLabel',blanks)
set(gca,'XTick',1:21)
classes = dataset_params.classes;

for q = 1:length(classes)
  text(q-.1,-.02,classes(q),'Rotation',270+45,'FontSize',18,'FontWeight','Bold')
end

title('PASCAL VOC2007 Object Detection AP','FontSize',18, ...
      'FontWeight','Bold')
ylabel('AP','FontSize',18,'FontWeight','Bold');
legend('ESVM','ESVM+B','ESVM+B+M')

dt = [.127 .253 .005, .015, .107 .205 .23 .005 .021 .128 .014 .004 ...
      .122 .103 .101 .022 .056 .05 .12 .248];
ldpm = [.287 .51 .006 .145 .265 .397 .502 .163 .165 .166 .245 .05 ...
        .452 .383 .362 .09 .174 .228 .341 .384]

aps(:,4) = dt;
aps(:,5) = ldpm;
set(gcf,'PaperPosition',[0 0 16 4])
set(gcf,'PaperSize',[16 4]);
imfiler = sprintf(['%s/newbars.pdf'],dataset_params.localdir);
print(gcf,imfiler,'-dpdf');

imfiler = sprintf(['%s/res_table2.tex'],dataset_params.localdir);
fid = fopen(imfiler,'w');
fprintf(fid,['\\begin{table}\n\\begin{center}\n\\begin{tabular}{|l||' ...
             'c|c|c|c|c|}\n\\hline\n'],tstr);
fprintf(fid,'Type & ESVM & ESVM-B & ESVM-M & DT & LDPM\\\\\n');
fprintf(fid,'\\hline\n');
c = dataset_params.classes;
for i = 1:20
  fprintf(fid,'%s & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n',...
          c{i},aps(i,1),aps(i,2),aps(i,3),aps(i,4),aps(i,5));
end
fprintf(fid,'\\hline\n');
fprintf(fid,'mAP & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n',...
        mean(aps(:,1)),mean(aps(:,2)),mean(aps(:,3)),mean(aps(:, ...
                                                  4)),mean(aps(:,5)));
fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n\\end{center}\n\\caption{hi}\n\\end{table}');
fclose(fid);

imfiler = sprintf(['%s/res_table.tex'],dataset_params.localdir);
fid = fopen(imfiler,'w');
tstr = '{c';

for i = 1:21
  tstr = [tstr '|c'];
end
tstr = [tstr '}'];


fprintf(fid,'\\begin{table}\n\\begin{center}\n\\begin{tabular}%s\n\\hline\n',tstr);
fprintf(fid,'Class & %s ',c{1});
for i = 2:20
  fprintf(fid,' & %s ',c{i});
end

fprintf(fid,' & mAP \\\\ \n');
fprintf(fid,'\\hline\n');

fprintf(fid,'MODE1 & %.2f ',aps(1,1));
for i = 2:20
  fprintf(fid,' & %.2f',aps(i,1));
end
fprintf(fid,'& %.2f\\\\ \n',mean(aps(:,1)));

fprintf(fid,'MODE2 & %.2f ',aps(1,2));
for i = 2:20
  fprintf(fid,' & %.2f ',aps(i,2));
end
fprintf(fid,' & %.2f\\\\ \n',mean(aps(:,2)));

fprintf(fid,'MODE3 & %.2f ',aps(1,3));
for i = 2:20
  fprintf(fid,' & %.2f ',aps(i,3));
end
fprintf(fid,'& %.2f \\\\ \n',mean(aps(:,3)));
fprintf(fid,'\\end{tabular}\n\\end{center}\n\\caption{hi}\n\\end{table}');

fclose(fid);

keyboard

function myplot(x,y,col,lstring,lwidth)
N = length(x);
N = min(N,5000);
plot(x(1:N),y(1:N),col,lstring,lwidth)
