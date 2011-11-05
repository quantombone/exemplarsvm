function ccc = kp_graph(cls)
%Show the kanade-perona graph which shows the performance of our
%system as a function of the number of exemplars.


ccc = '';
if ~exist('cls','var')
  
  % cls = {'bicycle'};
  % %cls = {'sheep'};
  % %cls = {'horse'};
  % cls = {'cow'};
  % cls = {'bus'};
  % cls = {'cat'};
  % cls = {'diningtable'};
  % cls = {'tvmonitor'};
  % cls = {'person'};
  % cls = {'motorbike'};
  
  
  cls={...
%    'aeroplane'
    'bicycle'
%    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    %'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    %'tvmonitor'
      };

  
  %cls = {'horse','bus','sofa','train','dog','boat', ...
  %       'aeroplane'};
  figure(34)
  clf
  for i = 1:length(cls)
    cs{i} = kp_graph(cls{i});
  end
  
  lens = cellfun(@(x)length(x),cs);
  cs = cs(lens>0);
  legend(cs,'Location','Southeast');
  %set(gcf,'LineWidth',4)
  h=title('PASCAL VOC AP vs. Number of Exemplars');
  set(h,'FontSize',16);
  print(gcf,'-depsc2','/nfs/baikal/tmalisie/nn311/kplog2.eps');
  return;
end
files = ['/nfs/baikal/tmalisie/nn311/VOC2007/results/N=%d.g-100-' ...
         '12.exemplar-svm.%s-calibrated_test_results.mat'];
         

for i = 1:1500
  try
    r = load(sprintf(files,i,cls));
    aps(i) = r.results2.apold;
  catch
    break
  end
end

%figure(34)
%clf

if aps(end) < .16
  return;
end
ccc = cls;
%h=plot(aps);
h=plot(100*(1:length(aps))/length(aps),aps);
set(h,'LineWidth',4)
xlabel('% exemplars','FontSize',18)
ylabel('AP','FontSize',18)
grid on
drawnow
hold all;

%filer = sprintf('/nfs/baikal/tmalisie/nn311/VOC2007/kp_graph/%s.png',cls);
%print(gcf,'-dpng',filer)
