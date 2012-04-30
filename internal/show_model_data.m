function Icur = show_model_data(model, K, show)
%Shows the model's positives and negatives as crops in one large
%matrix
%  Inputs:
%     model: the model
%     K: the KxK grid which shows positive and negatives
%  Outputs:
%    The image which shows results (if specified, doesn't do the showing)

if ~exist('K','var')
  K = 5;
end

if ~exist('show','var')
  show = 1;
end

r = model.models{1}.w(:)'*model.models{1}.x- ...
               model.models{1}.b;
model.models{1}.bb(:,end) = r;
[aa,bb] = sort(r,'descend');

Icur = esvm_show_det_stack(model.models{1}.bb(bb,:),model.data_set, ...
                           K,K,model.models{1});
if numel(model.models{1}.svxs) > 0
  r = model.models{1}.w(:)'*model.models{1}.svxs - model.models{1}.b;
  [aa,bb] = sort(r,'descend');
  model.models{1}.svbbs(:,end) = r;
  Icur2 = esvm_show_det_stack(model.models{1}.svbbs(bb,:), ...
                              model.data_set, K,K, ...
                              model.models{1});
  Ipad = zeros(size(Icur,1),10,3);
  Ipad(:,:,1) = 1;
  Icur = cat(2,Icur,Ipad,Icur2);
end

if show == 0
  return;
end
%else show
objective = evaluate_obj(model.models{1});
figure(39)
imagesc(Icur)
title(sprintf('%s: objective=%.5f',model.model_name,objective),'FontSize',20);
drawnow
snapnow

if nargout == 0
  Icur = [];
end
