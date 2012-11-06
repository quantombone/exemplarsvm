function model = hog_to_models(dataset,res)

%model = construct_scene_models(dataset(1),res);
params = esvm_get_default_params;

%Do not do flipping during detection
params.detect_add_flip = 0;

%Take at most one detection per image
params.detect_max_windows_per_exemplar = 1;

%Maximum image size, so that we get big images
params.max_image_size = 200;

for i = 1:length(dataset)
  I = toI(dataset(i));
  s1 = size(I,1);
  I = imresize_max(I,110);
  s2 = size(I,1);
  x2 = esvm_features(I,8);
  s = size(x2);
  
  model.models{i}.hg_size = s;
  model.models{i}.x = x2(:);
  
  s = s(1:2);
  bb(i,1:4) = ([1 1 s(2)*8 s(1)*8] + 8)*(s1/s2);
  bb(i,12) = 0;
  bb(i,11) = i;
  model.models{i}.bb = bb(i,:);
  model.models{i}.w = x2-mean(x2(:));
  model.models{i}.b = 0;
  model.models{i}.cls = sprintf('scene=%d',i);

  x(:,i) = x2(:);

end
  
%m = model.models{1};
%model = rmfield(model,'data_set');

subinds = get_subinds(res,size(x2));
lambda = .01;
c = (lambda*eye(length(subinds))+...
       res.c(subinds, ...
             subinds));

w2 = c\bsxfun(@minus,x,res.mean(subinds));

%cinv = inv(c);
%tic
%w2 = bsxfun(@minus,cinv*x,cinv*res.mean(subinds));
%toc

for i = 1:size(x,2)

  fprintf(1,'.');
  
  w = w2(:,i);
  [w,b] = rescale_w(w,x(:,i),res.mean(subinds));

  model.models{i}.w = reshape(w,model.models{i}.hg_size);
  model.models{i}.b = b;
  
  
  model.models{i}.center = model.models{i}.bb(1:4);
  model.models{i}.curc = model.models{i}.center;
  %m.curc = [1 1 640 480];%m.center(1:4)
  
end

model.data_set = dataset;
model.params = params;