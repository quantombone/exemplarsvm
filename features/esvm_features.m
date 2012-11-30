function x = esvm_features(I, sbin)
%Return the current feature function

if nargin == 0
  x = 31;
  return
end


%x = esvm_features2(I,sbin);
%return;

x = features_pedro(I,sbin);
