function Icur = show_model_data(model, K)
%Shows the model's positives and negatives as crops in one large matrix

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
figure(1)
clf
imagesc(Icur)
title(sprintf('%s',model.model_name),'FontSize',20);
drawnow
snapnow
