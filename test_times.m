mall = repmat(models2,1,10);
nex = 2:50:length(mall);
times = zeros(size(nex));
for i = 1:length(nex)
  r = randperm(length(mall));  
  cur = mall(r(1:nex(i)));
  starter = tic;
  apply_voc_exemplars(cur);
  times(i)=toc(starter);
  figure(1)
  clf
  plot(nex,times,'r.');
  hold on;
  plot(nex,times2,'b.');
  drawnow
end
  