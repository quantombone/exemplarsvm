function infer_from_ranks
%do inference from rank data

N = 2000;
K = 30;
pmistake = 0.0;

if 1
thetas = linspace(0,2*pi,K+1);
thetas = thetas(1:end-1);
x = [cos(thetas); sin(thetas)];
% x = randn(2,K);
% d = distSqr_fast(x,x);
% [aa,bb] = sort(d(1,:));
% x = x(:,bb);

%d is the real distance matrix
d = distSqr_fast(x,x);






%the ids are elements A B C where d_AB > d_AC
ids = zeros(3,2*N);
y = ones(1,2*N);
for i = 1:2*N
  r = randperm(K);
  r = r(1:2);
  r2 = randperm(K);
  r = [r r2(1)];
  if d(r(1),r(2)) < d(r(1),r(3))
    r = [r(1) r(3) r(2)];
  end
  

  ids(:,i) = r;
end

ids2 = ids(:,(N+1):end);
y2 = y(N+1:end);
ids = ids(:,1:N);
y = y(1:N);

subset = find(rand(size(y))<pmistake);
y(subset) = y(subset)*-1;
% if rand < pmistake
%   %%mess up some times
%   %r = [r(1) r(3) r(2)];
%   y(i) = -1;
% end

xs = zeros(K*K,N);
for i = 1:N
  deltaab = zeros(K,K);
  deltaab(ids(1,i),ids(2,i))=1;
  
  deltaac = zeros(K,K);
  deltaac(ids(1,i),ids(3,i))=1;  
  xs(:,i) = deltaac(:)-deltaab(:);
end



wpos = 1.0;
model = liblinear_train(y', -sparse(xs)', ...
                        sprintf(['-s 3 -B -1 -c 10' ...
                    ' -w1 %.3f -q'],wpos));
if y(1) == -1
  model.w = model.w*-1;
end
d2=(reshape(model.w,K,K));


figure(1)
subplot(2,2,1)
imagesc(d)
title('original distance matrix')
subplot(2,2,2)
imagesc(d2)
title('recovered distance matrix')

i1=(sub2ind(size(d),ids(1,:),ids(2,:)));
i2=(sub2ind(size(d),ids(1,:),ids(3,:)));


j1=(sub2ind(size(d),ids2(1,:),ids2(2,:)));
j2=(sub2ind(size(d),ids2(1,:),ids2(3,:)));


% for i = 1:N
%   for j = 1:N
%     for k = 1:N
      
%     end
%   end
% end

drawnow
end

%mean(d(i1) - d(i2)>=0)
mean(d2(i1) - d2(i2)>=0)

%mean(d(j1) - d(j2)>=0)
mean(d2(j1) - d2(j2)>=0)

subplot(2,2,3)
[U,W,V] = svd(d);
plot(U(:,2),U(:,3),'r.');
title('original embedding')


subplot(2,2,4)
[U,W,V] = svd(d2);
%Y = mdscale(d2,2);
plot(U(:,1),U(:,2),'r.');
title('recovered embedding')
