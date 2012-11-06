function model = learnGaussianTriggs(data_set, cls, covstruct_full, ...
                                     add_flips, hg_size,A)
% Learn a DalalTriggs template detector, with latent updates,
% perturbed assignments (which help avoid local minima), and
% ability to use different features.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

params = esvm_get_default_params;

if ~exist('add_flips','var')
  add_flips = 1;
end

if exist('hg_size','var') && length(hg_size) > 0
  params.hg_size = hg_size;
  params.init_params.hg_size = hg_size;
end


fprintf(1,'Initializing by warping positives\n');
fprintf(1,'add_flips is %d\n',add_flips);
params.dt_initialize_with_flips = add_flips;
params.dt_pad_factor = .1;
fprintf(1,'PAD factor is %f\n',params.dt_pad_factor);

model = esvm_initialize_dt(data_set,cls,params);
fprintf(1,'done initing\n');
if length(model) == 0
  return;
end

model.models{1}.cls = cls;

s = model.models{1}.hg_size(1:2);
c = [0 0 s(1) s(2)];
c2 = c + [-s(2)*params.dt_pad_factor -s(1)*params.dt_pad_factor ...
          s(2)*params.dt_pad_factor s(1)*params.dt_pad_factor];
model.models{1}.center = c2;
model.models{1}.curc = c;

hg_size = model.models{1}.hg_size;
subinds = get_subinds(covstruct_full,hg_size);

x = mean(model.models{1}.x,2);



if exist('A','var') && length(A)>0
  w2 = A*(x-covstruct_full.mean(subinds));
else
  lambda = .01;
  w2 = (lambda*eye(length(subinds))+...
        covstruct_full.c(subinds, ...
                         subinds))\(x- ...
                                    covstruct_full.mean(subinds));
end

% if params.dt_initialize_with_flips
%   w2 = w2*0+randn(size(w2));
%   for q = 1:20
%     scores = w2(:)'*model.models{1}.x;
%     [~,ind] = max(reshape(scores,2,[]),[],1);

%     newinds = 2*(1:(numel(scores)/2))+(ind-2);
%     x = mean(model.models{1}.x(:,newinds),2);
%     w2 = (lambda*eye(length(subinds))+ ...
%           covstruct_full.c(subinds, ...
%                            subinds))\(x-covstruct_full.mean(subinds));
%     s=w2(:)'*x;
%     s
%   end

% end

if 0
  [covstruct,subinds,hg_full] = subcov(covstruct_full, ...
                                       model.models{1}.hg_size);
  
  %goods = find(covstruct.evals>.0001);
  %goods = find(covstruct.evals>.0000000001);
  %goods = find(covstruct.evals>.01);
  covstruct.evals = max(0,covstruct.evals);
  %fprintf(1,'choosing %d out of %d eigenvectors\n',length(goods),length(covstruct.evals));
  V = covstruct.evecs;%(:,goods);
  D = (covstruct.evals);%(goods));
  
  newD = diag(D./(.00001+D.^2));
  
  newmat = V*newD*V';
  w = newmat*(x(:)-covstruct.mean);
  
else
  w = w2;
end


% cpos=cov(model.models{1}.x');
% w = (cpos+.0001*eye(size(cpos)))\(mean(model.models{1}.x,2));
% w = w - mean(w(:));

% [V1,D1] = eig(covstruct_full.c(subinds,subinds));
% goods = diag(D1)>.001;
% V1 = V1(:,goods);
% D1 = D1(goods,goods);

% [V2,D2] = eig(cpos);
% [aa,bb] = sort(diag(D2),'descend');
% goods = bb(1:10);



% V2 = V2(:,goods);
% D2 = D2(goods,goods);

% Akernel = .5*(V1*diag(1./diag(D1))*V1'+V2*diag(1./diag(D2))*V2');
% [V,D] = eig(Akernel);
% D(D<.0001) = 0;
% model.models{1}.Akernel = D^.5*V';

% model.params.max_models_before_block_method = 0;
% model.models{1}.params.max_models_before_block_method = 0;



mean_x = x;
for qqq = 1:1
w = reshape(w,model.models{1}.hg_size);
model.models{1}.w = w;
model.models{1}.b = 0;
model.models{1}.w = model.models{1}.w;

[alphas] = inv([w(:)'*mean_x 1; w(:)'*covstruct_full.mean(subinds) 1])*[1 -1]';
model.models{1}.w = alphas(1)*model.models{1}.w;
model.models{1}.b = -alphas(2);

return;

pset = split_sets(data_set,cls,0);
[dets,savedets] = applyModel(pset,model,.001);

fb = cat(1, ...
         savedets ...
         .final_boxes{:});
fb = esvm_nms(fb);
[aa,bb] = sort(fb(:,end),'descend');
fb = fb(bb,:);

results = VOCevaldet(pset,esvm_nms(fb),cls,.5);
results = VOCevaldet(pset,fb,cls,.5);

goods = find(results.is_correct==1); % & results.prec>.1);
bads = find(results.is_correct==0);
[target_id,target_x] = ...
    esvm_reconstruct_features(fb(goods,:),model,pset,10000);

mean_x = mean(target_x,2);

[target_id,bad_x] = ...
    esvm_reconstruct_features(fb(bads,:),model,pset,500);

w = (newmat)*(mean_x - covstruct.mean);



end

