function [evecicons] = show_cov(res,K)
% Compute eigenvalues and eigenvectors of the HOG covariance matrix
% estimated from the fields of the input [res].  If 3 outputs are
% specified, then returns a stack of eigenvector images by performing
% eigendecomposition of the covariance matrix

%Compute top 100 eigenvector images, which will viewable by using
%montage
fun = @HOGpicturemax;
for a = 1:K
  evec = res.evecs(:,a);
  evec = evec - mean(evec(:));
  evecicons(:,:,:,a) = ...
      pad_image(cat(2,pad_image(jettify(fun(reshape(evec, ...
                                                  res.hg_size))),5), ...
                    pad_image(jettify(fun(reshape(-evec, ...
                                                  res.hg_size))),5)),2,[1 ...
                    0 0]);
  fprintf(1,'.');
end

if nargout == 0
  %Display everything
  montage(evecicons)
end

