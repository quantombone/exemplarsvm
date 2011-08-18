function kp_graph(cls)

if ~exist('cls','var')
  cls = 'bus';
  cls = 'horse';
  cls = {'horse','bus','sofa','train','dog','boat', ...
         'aeroplane'};
  for i = 1:length(cls)
    kp_graph(cls{i});
  end
  return;
end
files = ['/nfs/baikal/tmalisie/nn311/VOC2007/results/N=%d.g-100-' ...
         '12.exemplar-svm.%s-calibrated_test_results.mat'];
         

for i = 1:100
  r = load(sprintf(files,i,cls));
  aps(i) = r.results2.apold;
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
