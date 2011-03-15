function res = mysvmpredict(newvals,svm_model)

ddd = distSqr_fast(newvals,svm_model.SVs');
ddd = exp(-ddd*svm_model.Parameters(4));

ddd = bsxfun(@times,svm_model.sv_coef',ddd);
res = sum(ddd,2) - svm_model.rho;
