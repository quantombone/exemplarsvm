function reconstruct_X(X,y,ids)
%% Here we try to take a bunch of X's (a subset of categories from
%% SUN397 and apply the reconstructive potential matrix
[v,d,mu] = mypca(X,200);

%choose phone booths and control towers
targety = [275 111];
[aa,bb] = ismember(y,targety);
goods = find(aa==1);
others = find(y<=0);
finaly = y([goods; others]);
X2 = X(:,others);
X = X(:,goods);
X = cat(2,X,X2);
fprintf(1,'doing pca\n');

X = v'*bsxfun(@minus,mu,X);

ids = ids([goods; others]);

W = X;
N = size(X,2);
d = distSqr_fast(X,X);
A = exp(-.005*d/mean(d(:)));
A = (A+A')/2;
if 0
for i=1:size(A,1)
  [aa,bb] = psort(-A(i,:)',5);
  A(i,bb)=1;
  A(i,i) = 1;
end
end

cx = repmat(finaly(:),1,length(finaly));
cy = cx';
A(cx==cy) = 1;
A(cx~=cy) = -1;

%A = A*0-1;
%A(find(speye(size(A))))=1;

shower = zeros(100,100,3);
X(end+1,:) = 1;
for q = 1:2000
[aa,bb] = sort(A(1,:),'descend');

F = size(X,1);
lambda = 10;
tic
W = pinv((X*X'+lambda*eye(F))')*X*A';
%K = X'*X;
%E = inv((K'*K+lambda*K')')*A*K';
%W = X*E;
toc

A = W'*X;


for i = 1:size(A,1)
  [aa1,bb1] = sort(A(i,:),'descend');
  %A(i,:) = 0;
  [aaa,bbb]= sort(bb1);
  %A(i,:) = 1./(bbb);
end

A = (A+A')/2;
%A = A + .1*randn(size(A));
%A = A - diag(diag(A));
%A = A + eye(size(A));

if 0
cx = repmat(finaly(:),1,length(finaly));
cy = cx';
A(cx~=cy) = -1;
end

figure(1)
clf

if mod(q,100)==0
  shower = create_grid_res(ids(bb),4);
end

subplot(1,3,1)
clf
imagesc(shower)

subplot(1,3,2)
plot(A(2,:))
subplot(1,3,3)
imagesc(A)
drawnow

end

