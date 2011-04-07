function res = mysvmpredict(newvals,svm_model)
%Apply SVM decision rule for libSVM trained models

%if these are present, it is because I already computed the
%decision boundary -- this means the model is linear!
if isfield(svm_model,'w') && isfield(svm_model,'b')
  res = svm_model.w'*newvals - svm_model.b;
  return;
end

%perform kernel expansions, ...
ddd = distSqr_fast(newvals,svm_model.SVs');
ddd = exp(-ddd*svm_model.Parameters(4));

ddd = bsxfun(@times,svm_model.sv_coef',ddd);
res = sum(ddd,2) - svm_model.rho;
