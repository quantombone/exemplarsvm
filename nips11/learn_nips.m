function learn_nips(models)
%Learn the graph model here
xs = cellfun2(@(x)x.model.x,models);
X=[xs{:}];
%% add bias term
X(end+1,:) = 1;
%W = bsxfun(@minus,mean(X,2),X);

R = exp(-.01*distSqr_fast(X,X));

R = R + diag(diag(R))*3;

for qqq = 1:100
  
  if 1
    for i = 1:size(R,1)
      [aa,bb] = sort(R(i,:),'descend');
      R(i,bb(21:end)) = -1;
      R(i,:) = max(0,R(i,:));
      R(i,:) = R(i,:) / sum(R(i,:));
      
      % figure(1)
      % for q = 1:16
      %   subplot(4,4,q)
      %   imagesc(get_exemplar_icon(models,bb(q)))
      % end
      % pause
    end
  end
%% LEARN W FIRST

K = X'*X;

%objective = @(x)norm(reshape(x,size(K))*K-R)^2+.001*norm(x)^2;

lambda = 100;
%A=pinv(K'*K + lambda*K)*R*K;

W=inv(X*X'+lambda*eye(size(X,1),size(X,1)))*X*R;
A = W'*pinv(X');

%Rold = R;
R = (A*K)';

%R2 = X'*W;

%R = R + R';
R = max(R,0);
for i = 1:size(R,1)
  R(i,:) = R(i,:) / sum(R(i,:));
end

for i = 13:13 %1:size(A,1)
  [aa,bb] = sort(R(i,:),'descend');
  figure(1)
  for q = 1:16
    subplot(4,4,q)
    imagesc(get_exemplar_icon(models,bb(q)))
    title(num2str(aa(q)))
  end
  drawnow
  pause(.1)
end


end