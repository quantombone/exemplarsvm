function [w,resc] = build_bc(x,cls,target,res)
%r = randperm(size(x,2));

y = double(ismember(cls,target));
y(y==0) = -1;


%tic
%c = cov(x(:,r(1:20000))');
%toc

subinds = get_subinds(res,[8 8 31]);
c = res.c(subinds,subinds);

% [v,d] = eig(c);
% [aa,bb] = sort(diag(d),'descend');
% v = v(:,bb);
% d = d(bb,bb);
% d = diag(d);

% d2 = d*0;
% d2(1:300) = 1./d(1:300);
% cinv = v*diag(d2)*v';

uc = unique(cls);
[~,ids] = ismember(cls,uc);
counts = hist(ids,1:length(uc));
[alpha,beta] = sort(counts,'descend');

for i = 1:200
  resc{i} = uc{beta(i)};
  m(:,i) = mean(x(:,ids==beta(i)),2);
end

m2 = bsxfun(@minus,m,res.mean(subinds));

w = (c+.01*eye(size(c)))\m2;


%diffs = (mean(x(:,y==1),2) - mean(x(:,y==-1),2));;
%w = (c+.01*eye(size(c)))\diffs;
%w = cinv*diffs;
%w = [w; w'*mean(x(:,y==-1),2)];

w(end+1,:) = 0;
for i = 1:size(w,2)
  [w(1:end-1,i),w(end,i)] = rescale_w(w(1:end-1,i),m(:,i),res.mean(subinds));%mean(x(:,y==-1),2));

end
%w(1:end-1)'*mean(x(:,y==1),2)-b
%w(1:end-1)'*res.mean(subinds)-b


%params.regularizer = .01*eye(length(w));
%params.regularizer(end) = 0;
%params.c = [w*0];
%[w,svmobj] = svmlsq(y,double(x),w,params);