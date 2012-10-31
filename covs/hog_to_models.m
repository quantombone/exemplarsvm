function model = hog_to_models(dataset,x,res)

model = construct_scene_models(dataset.image_files(1),res);

I = toI(dataset.image_files(1));
I2 = imresize_max(I,110);
x2 = esvm_features(I2,8);

m = model.models{1};
model = rmfield(model,'data_set');

subinds = get_subinds(res,size(x2));
lambda = .01;
c = (lambda*eye(length(subinds))+...
       res.c(subinds, ...
             subinds));
cinv = inv(c);
tic
w2 = bsxfun(@minus,cinv*x,cinv*res.mean(subinds));
toc

for i = 1:size(x,2)
  fprintf(1,'.');
  m = model.models{1};
  m.hg_size = size(x2);
  m.bb(11) = i;
  m.cls = sprintf('scene=%d',i);
  %w
  %w = cinv*(x(:,i) - res.mean(subinds));
  m.w = w2(:,i);
  [m.w,m.b] = rescale_w(m.w,x(:,i),res.mean(subinds));
  m.w = reshape(m.w,m.hg_size);
  ms{i} = m;
end

model.models = ms;
model.data_set = dataset.image_files;