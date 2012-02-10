function model = esvm_latent_update_dt(model, params)
% Updates positives with latent placements which pass os threshold test
% 
% INPUTS:
% models: last iteration's model whose xs and bbs will be updated
% params [optional]: the parameters of the model
%
% OUTPUTS:
% models: the updated models
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if exist('params','var')
  model.params = params;
elseif isfield(model, 'params')
  params = model.params;
else
  params = esvm_get_default_params;
  model.params = params;  
end

if length(model.models) > 1
  fprintf(1,'WARNING: latent update has more than one model\n');
end

if length(params.localdir)>0
  CACHE_FILE = 1;
else
  CACHE_FILE = 0;
end


model_name = [model.model_name '-L'];

cache_dir =  ...
    sprintf('%s/models/',params.localdir);

cache_file = ...
    sprintf('%s/%s.mat',cache_dir,model_name);

if CACHE_FILE ==1 && fileexists(cache_file)
  model = load(cache_file);
  model = model.model;
  return;
end

% results_directory = ...
%     sprintf('%s/models/%s/',params.dataset_params.localdir, ...
%             model_name);

% if CACHE_FILE==1 && ~exist(results_directory,'dir')
%   fprintf(1,'Making directory %s\n',results_directory);
%   mkdir(results_directory);
% end


filer = cache_file;

%NOTE BUG (TJM): no filerlock locking here

p = params;%esvm_get_default_params;
p.detect_save_features = 1;
p.detect_keep_threshold = -1.0;
p.detect_exemplar_nms_os_threshold = 1.0;
p.detect_max_windows_per_exemplar = 300;

curx = cell(0,1);
curbb = cell(0,1);

data_set = model.data_set;

potentialx = cell(0,1);
potentialbb = cell(0,1);
fprintf(1,'esvm_latent_update_dt: starting updates\n');
for j = 1:length(data_set)  
  
  obj = {data_set{j}.objects};
  if length(data_set{j}.objects) == 0
    continue
  end
  
  I = toI(data_set{j});
  rs = esvm_detect(I, model, p);

  if size(rs.bbs{1})~=0
    rs.bbs{1}(:,11) = j;
    gt_bbs = cat(1,data_set{j}.objects.bbox);
    for k = 1:size(gt_bbs,1)
      %get overlaps, and keep ones above threshold
      os = getosmatrix_bb(rs.bbs{1},gt_bbs(k,:));
      goods = find(os >= params.latent_os_thresh);
      if length(goods) >=1
        curx{end+1} = rs.xs{1}{goods(1)};
        curbb{end+1} = rs.bbs{1}(goods(1),:);
        
        potentialx{end+1} = cat(2,rs.xs{1}{goods});
        potentialbb{end+1} = rs.bbs{1}(goods,:);
        
        fprintf(1,'+');
      else
        fprintf(1,'-');
      end
    end
  end
end

USE_ALL_POTENTIALS = 0;

if USE_ALL_POTENTIALS == 1
  model.models{1}.x = cat(2,potentialx{:});
  model.models{1}.bb = cat(1,potentialbb{:});
  [~,bb] = sort(model.models{1}.w(:)'*model.models{1}.x,'descend');
  model.models{1}.x = model.models{1}.x(:,bb);
  model.models{1}.bb = model.models{1}.bb(bb,:);
else
  
  best_model = model.models{1};
  best_score = evaluate_obj(best_model);
  
  %now we are onto finding the best assignments  
  %we always do first iterations, because it is greedy
  for z = 1:max(1,params.latent_perturb_assignment_iterations)
    fprintf(1,' ---###--- Latent trying configuration %d\n',z);
    scores = cellfun2(@(x)best_model.w(:)'*x-best_model.b, ...
                      potentialx);
    clear ass
    for i = 1:length(scores)
      [aa,bb] = sort(scores{i},'descend');
      r = randperm(min(3,length(bb)));
      ass(i) = bb(r(1));
    end
    
    %First iteration is greedy (which is the classical latent update step)
    if (z == 1)
      ass = ass*0+1;
    end
    
    curx = cellfun(@(x,y)x(:,y),potentialx,num2cell(ass), ...
                   'UniformOutput',false);
    curx = cat(2,curx{:});
    
    curbb = cellfun(@(x,y)x(y,:),potentialbb,num2cell(ass), ...
                    'UniformOutput',false);
    curbb = cat(1,curbb{:});
    
    curm = best_model;
    curm.x = curx;
    curm.bb = curbb;
    curm = params.training_function(curm);
    cur_score = evaluate_obj(curm);
    if cur_score < best_score
      best_model = curm;
      
      fprintf(1,'+ Good objective = %.5f, old = %.5f\n',cur_score,best_score);
      best_score = cur_score;
    else
      fprintf(1,'- Bad  objective = %.5f, old = %.5f\n',cur_score,best_score);
    end
  end
  
  models{1} = best_model;
end

%Update name with proper suffix being concatenated
models{1}.model_name = [model_name];

fprintf(1,'Got latent updates for %d examples\n',size(models{1}.bb,1));

if CACHE_FILE == 1
  save(filer,'models');
end

% if (m.params.dump_images == 1) || ...
%       (m.params.dump_last_image == 1 && ...
%        m.iteration == m.params.train_max_mine_iterations)

%   imwrite(Isv1,sprintf('%s/%s.%d_iter_I=%05d.png', ...
%                     m.params.localdir, m.model_name, ...
%                     m.identifier, m.iteration), 'png');
% end

if params.display == 1    
  [aa,bb] = sort(models{1}.w(:)'*models{1}.x,'descend');
  Icur = esvm_show_det_stack(models{1}.bb(bb,:),model.data_set, ...
                             10,10,models{1});
  
  [aa,bb] = sort(models{1}.w(:)'*models{1}.svxs,'descend');
  Icur2 = esvm_show_det_stack(models{1}.svbbs(bb,:), ...
                              model.data_set, 10,10, ...
                              models{1});
  Ipad = zeros(size(Icur,1),10,3);
  Ipad(:,:,1) = 1;
  Icur = cat(2,Icur,Ipad,Icur2);
  figure(1)
  clf
  imagesc(Icur)
  title('Positives and Negatives After Latent Update','FontSize',20);
  drawnow
  snapnow
  if params.dump_images == 1 && length(params.localdir) > 0
    filerpng = [filer '.png'];
    if ~fileexists(filerpng)
      imwrite(Icur,filerpng);
    end
  end
end

