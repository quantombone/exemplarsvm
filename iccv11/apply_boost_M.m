function r = apply_boost_M(x, boxes, M)
%% Apply the multiplexer matrix M which boosts the scores of a
%% window based on its friends and their scores embedded in the
%% context feature vector x
if prod(size(x))==0
  r = zeros(0,1);
  return;
end
exids = boxes(:,6);
exids(boxes(:,7)==1) = exids(boxes(:,7)==1) + size(x,1)/2;
r = zeros(1,size(boxes,1));

for i = 1:size(boxes,1)
  if isfield(M,'C1')
    r(i) = (x(:,i)'*M.C1*x(:,i));
  elseif isfield(M,'svm_model')
    % a non-model, means no firings, so return null
    if length(M.svm_model{exids(i)}) > 0
      r(i) = mysvmpredict(x(:,i),M.svm_model{exids(i)});
    else
      r(i) = -2;
    end
  else

    r(i) = (M.w{exids(i)}'*x(:,i) + sum(x(:,i)))-M.b{exids(i)};
    %r(i) = (M.w{exids(i)}'*x(:,i))-M.b{exids(i)};
  end
end
