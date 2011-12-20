function hogpic = jettify(hogpic)
%Turn a matrix which can be viewed via imagesc into an image using
%the same jet scheme, but now the result can be written to a file
NC = 200;
colorsheet = jet(NC);
dists = hogpic(:);    
dists = dists - min(dists);
dists = dists / (max(dists)+eps);
dists = round(dists*(NC-1)+1);
colors = colorsheet(dists,:);
hogpic = reshape(colors,[size(hogpic,1) size(hogpic,2) 3]);