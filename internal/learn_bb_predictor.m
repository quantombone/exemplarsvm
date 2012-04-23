function model2=learn_bb_predictor(model)
%Learn a regressor which maps raw input features to the appropriate
%bounding box
r = model.models{1}.w(:)'*model.models{1}.x - model.models{1}.b;
B = model.models{1}.resc';

% S = repmat(model.models{1}.w(:),1,size(model.models{1}.x, ...
%                                              2)).* ...
%     model.models{1}.x;

S = model.models{1}.x;
S(end+1,:)=1;
lambda = .01;

% tic
% A = (B)*S'*pinv(S*S' + lambda*eye(size(S,1)));
% toc

A = ((S*S'+lambda*eye(size(S,1)))\(S*B'))';

newB = A*S;

model2 = model;
model2.models{1}.A = A;
hg_size = model.params.init_params.hg_size;
center = [0 0 10*hg_size(1) 10*hg_size(2)];
model2.models{1}.center = center;

return;

% osmat = getosmatrix_bb(newB',B');
% mean(diag(osmat))

% osmat2 = getosmatrix_Bb(B',repmat(center,size(B,2),1));
% mean(diag(osmat2))



[aa,bb] = sort(r,'descend');
B = B(:,bb);
newB = newB(:,bb);

for i = 1:size(B,2)
  figure(1)
  clf
  imagesc(zeros(80,80,3))
  plot_bbox(B(:,i)','GT',[0 1 0])
  plot_bbox(newB(:,i)','P',[1 0 0])
  axis image
  title(num2str(i))
  pause
end
