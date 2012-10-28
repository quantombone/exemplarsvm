function [res2,subinds,hg_full] = subcov(res,hg_size)
%hg_size = model.models{1}.hg_size;
fprintf(1,'Finding sub-covariance matrix of size%dx%d\n', ...
        hg_size(1),hg_size(2));
tic
hg_full = zeros(res.hg_size(1), res.hg_size(2), res.hg_size(3));

offset = [0 0];
offset = round((res.hg_size(1:2) - hg_size(1:2))/2);

hg_full(offset(1) + (1:hg_size(1)),...
        offset(2) + (1:hg_size(2)),:) = 1;

subinds = find(hg_full);

mu = res.mean(subinds);
c = res.c(subinds,subinds);

res2.c = c;
res2.mean = mu;
%return;

[v,d] = eig(c);
d = diag(d);
[aa,bb] = sort(d,'descend');
v = v(:,bb);
d = d(bb);

% figure(1)
% clf
% for q = 1:10
%   curv = v(:,q);
%   curv = curv - mean(curv(:));
%   subplot(3,3,q)
%   imagesc(HOGpicture(reshape(curv,hg_size)))
% end

res2.c = c;
res2.mean = mu;
res2.hg_size = hg_size;
res2.evals = d;
res2.evecs = v;
toc
