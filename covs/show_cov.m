function [evecicons,evals,evecs] = show_cov(res)
% Create visualizations of the HOG covariance matrix estimated from
% the fields of res.
% Return stack of eigenvector images by performing
% eigendecomposition of the covariance matrix

c = res.c;

%c = inv(c);
tic
[v,d] = eig(c);
d = diag(d);
[aa,bb] = sort(d,'descend');
v = v(:,bb);
d = d(bb);
toc

evals = d;
evecs = v;

%Compute top 100 eigenvector images, which will viewable by using
%montage
for a = 1:100
  evec = v(:,a);
  evec = evec - median(evec(:));
  evecicons(:,:,:,a) = ...
      pad_image(cat(2,pad_image(jettify(HOGpicture(reshape(evec, ...
                                                  res.hg_size))),5), ...
                    pad_image(jettify(HOGpicture(reshape(-evec, ...
                                                  res.hg_size))),5)),2,[1 ...
                    0 0]);
  fprintf(1,'.');
end

%Display everything
montage(evecicons)

