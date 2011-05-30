function models = rescale_ws(models)
%Here I am rescaling the outputs of the current classifiers on the
%mined set, and forcing it to behave nicely
for i = 1:length(models)
  r0 = models{i}.model.w(:)'*models{i}.model.nsv(:,1);
  rf = models{i}.model.w(:)'*models{i}.model.nsv(:,5);
  
  beta = inv([r0 -1; rf -1])*[1 0]';
  
  models{i}.model.w = models{i}.model.w*beta(1);
  models{i}.model.b = beta(2);
  plot(models{i}.model.w(:)'*models{i}.model.nsv-models{1}.model.b)

  
end
