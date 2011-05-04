function w2 = hog_normalize(w)
%normalize by norm, so that finding maximal dot product (like
%output of linear classifier) amounts to finding min-distance w's

if iscell(w)
  for i = 1:length(w)
    w2{i} = hog_normalize(w{i});
  end
  return;
end
s = sqrt(sum(w.^2,3));
w2 = w ./ repmat(s,[1 1 size(w,3)]);
w2(find(repmat(s,[1 1 size(w,3)])<.0000001)) = 1/sqrt(size(w,3));

