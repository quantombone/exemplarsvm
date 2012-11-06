function Icur = show_model_data(model, N, show)
%Shows the model's positives and negatives as crops in one large
%matrix
%  Inputs:
%     model: the model
%     N: the number of positives to show (also same number of
%     negatives) in a KxK grid where, K = ceil(sqrt(N)) 
%  Outputs:
%    The image (if specified, doesn't do the showing)
%  Tomasz Malisiewicz (tomasz@csail.mit.edu)

if ~exist('N','var')
  N = 25;
end

K = ceil(sqrt(N));

if ~exist('show','var')
  show = 1;
end

r = model.models{1}.w(:)'*model.models{1}.x- ...
               model.models{1}.b;
model.models{1}.bb(:,end) = r;
[aa,bb] = sort(r,'descend');

Icur = esvm_show_det_stack(model.models{1}.bb(bb,:),model.data_set, ...
                           K,K,model.models{1});
if isfield(model.models{1},'svxs') && numel(model.models{1}.svxs) > 0
  r = model.models{1}.w(:)'*model.models{1}.svxs - model.models{1}.b;
  [aa,bb] = sort(r,'descend');

  model.models{1}.svbbs = model.models{1}.svbbs(bb,:);
  model.models{1}.svbbs(:,end) = aa;
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

%objective = evaluate_obj(model.models{1});
objective = 1.0;
if nargout == 0
  clf
  imagesc(Icur)
  title(sprintf('cls=%s',model.models{1}.cls))
end
if ~isfield(model,'model_name')
  model.model_name = '';
end

%title(sprintf('%s: objective=%.5f',model.model_name,objective),'FontSize',20);
drawnow
snapnow

if nargout == 0
  Icur = [];
end
