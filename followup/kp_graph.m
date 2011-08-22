function kp_graph(cls)

if ~exist('cls','var')
  
  cls = {'bicycle'};
  %cls = {'sheep'};
  %cls = {'horse'};
  cls = {'cow'};
  cls = {'bus'};
  cls = {'cat'};
  cls = {'diningtable'};
  cls = {'tvmonitor'};
  cls = {'person'};
  cls = {'motorbike'};
  
  %cls = {'horse','bus','sofa','train','dog','boat', ...
  %       'aeroplane'};
  for i = 1:length(cls)
    kp_graph(cls{i});
  end
  return;
end
files = ['/nfs/baikal/tmalisie/nn311/VOC2007/results/N=%d.g-100-' ...
         '12.exemplar-svm.%s-calibrated_test_results.mat'];
         

for i = 1:1000
  try
    r = load(sprintf(files,i,cls));
    aps(i) = r.results2.apold;
  catch
    break
  end
end

figure(34)
clf

plot(aps,'r.-')
title(cls)
xlabel('# exemplars')
ylabel('AP')
grid on

filer = sprintf('/nfs/baikal/tmalisie/nn311/VOC2007/kp_graph/%s.png',cls);
print(gcf,'-dpng',filer)
