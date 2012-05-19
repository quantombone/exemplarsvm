function res2 = subcov(res,model)
hg_size = model.models{1}.hg_size;

hg_full = zeros(res.hg_size(1), res.hg_size(2), res.hg_size(3));

offset = [0 0];
offset = round((res.hg_size(1:2) - hg_size(1:2))/2);
offset
hg_full(offset(1) + (1:hg_size(1)),...
        offset(2) + (1:hg_size(2)),:) = 1;

subinds = find(hg_full);

mu = res.mean(subinds);
c = res.c(subinds,subinds);

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
