function x = esvm_features2(I, sbin)
%Return the current feature function

if nargin == 0
  x = 31;
  return
end

for i = 1:50
  I2 = I + randn(size(I))*.1;
  I2 = max(0,min(1.0,I2));
  xs(:,:,:,i) = features_pedro(I2, sbin);
end
x = mean(xs,4);

%x = features_raw(I,sbin);