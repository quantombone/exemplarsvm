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

[aa,bb] = sort(model.models{1}.w(:)'*model.models{1}.x,'descend');
Icur = esvm_show_det_stack(model.models{1}.bb(bb,:),model.data_set, ...
                           K,K,model.models{1});
if numel(model.models{1}.svxs) > 0
  [aa,bb] = sort(model.models{1}.w(:)'*model.models{1}.svxs,'descend');
  Icur2 = esvm_show_det_stack(model.models{1}.svbbs(bb,:), ...
                              model.data_set, K,K, ...
                              model.models{1});
  Ipad = zeros(size(Icur,1),10,3);
  Ipad(:,:,1) = 1;
  Icur = cat(2,Icur,Ipad,Icur2);
end

if nargout == 0
  return;
end
%else show
objective = evaluate_obj(model);
imagesc(Icur)
title(sprintf('%s: objective=%.3f',model.model_name,objective),'FontSize',20);
drawnow
snapnow
