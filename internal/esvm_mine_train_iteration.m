function [m] = esvm_mine_train_iteration(m, training_function)
%% ONE ITERATION OF: Mine negatives until cache is full and update the current
% classifier using training_function (do_svm, do_rank, ...). m must
% contain the field m.train_set, which indicates the current
% training set of negative images
% Returns the updated model (where m.mining_queue is updated mining_queue)
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

% Start wtrace (trace of learned classifier parameters across
% iterations) with first round classifier, if not present already
if ~isfield(m.model,'wtrace')
  m.model.wtrace{1} = m.model.w;
  m.model.btrace{1} = m.model.b;
end

if length(m.mining_queue) == 0
  fprintf(1,' ---Null mining queue, not mining!\n');
  return;
end

%If the skip is enabled, we just update the model
if m.mining_params.train_skip_mining == 0
  [hn, m.mining_queue, mining_stats] = ...
      esvm_mine_negatives({m}, m.mining_queue, m.train_set, ...
                     m.mining_params);

  m = add_new_detections(m, cat(2,hn.xs{1}{:}), cat(1,hn.bbs{1}{: ...
                   }));
else
  mining_stats.num_visited = 0;
  fprintf(1,'WARNING: train_skip_mining==0, just updating model\n');  
end
   
m = update_the_model(m, mining_stats, training_function);

if isfield(m,'dataset_params') && m.dataset_params.display == 1
  dump_figures(m);
end

function [m] = update_the_model(m, mining_stats, training_function)
%% UPDATE the current SVM, keep max number of svs, and show the results

if ~isfield(m,'mining_stats')
  m.mining_stats{1} = mining_stats;
else
  m.mining_stats{end+1} = mining_stats;
end

m = training_function(m);

% Append new w to trace
m.model.wtrace{end+1} = m.model.w;
m.model.btrace{end+1} = m.model.b;

% if (m.mining_params.dfun == 1)
%   r = m.model.w(:)'*bsxfun(@minus,m.model.svxs,m.model.x(:,1)).^2 - ...
%       m.model.b;
% else
%   r = m.model.w(:)'*m.model.svxs - m.model.b;
% end
% m.model.svbbs(:,end) = r;

function dump_figures(m)

% figure(1)
% clf
% show_cool_os(m)

% if (mining_params.dump_images == 1) || ...
%       (mining_params.dump_last_image == 1 && ...
%        m.iteration == mining_params.train_max_mine_iterations)
%   set(gcf,'PaperPosition',[0 0 10 3]);
%   print(gcf,sprintf('%s/%s.%d_iter=%05d.png', ...
%                     mining_params.final_directory,m.curid,...
%                     m.objectid,m.iteration),'-dpng'); 
% end

figure(2)
clf
Isv1 = esvm_show_det_stack(m,7);

imagesc(Isv1)
axis image
axis off
iter = length(m.model.wtrace)-1;
title(sprintf('Ex %s.%d.%s SVM-iter=%03d',m.curid,m.objectid,m.cls,iter))
drawnow
snapnow

if (m.mining_params.dump_images == 1) || ...
      (m.mining_params.dump_last_image == 1 && ...
       m.iteration == m.mining_params.train_max_mine_iterations)

  imwrite(Isv1,sprintf('%s/%s.%d_iter_I=%05d.png', ...
                    m.mining_params.final_directory, m.curid,...
                    m.objectid, m.iteration), 'png');
end

function m = add_new_detections(m, xs, bbs)
% Add current detections (xs,bbs) to the model struct (m)
% making sure we prune away duplicates, and then sort by score
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%First iteration might not have support vector information stored
if ~isfield(m.model, 'svxs') || isempty(m.model.svxs)
  m.model.svxs = [];
  m.model.svbbs = [];
end

m.model.svxs = cat(2,m.model.svxs,xs);
m.model.svbbs = cat(1,m.model.svbbs,bbs);

%Create a unique string identifier for each of the supports
names = cell(size(m.model.svbbs,1),1);
for i = 1:length(names)
  bb = m.model.svbbs(i,:);
  names{i} = sprintf('%d.%.3f.%d.%d.%d',bb(11),bb(8), ...
                             bb(9),bb(10),bb(7));
end
  
[unames,subset,j] = unique(names);
m.model.svbbs = m.model.svbbs(subset,:);
m.model.svxs = m.model.svxs(:,subset);

[aa,bb] = sort(m.model.w(:)'*m.model.svxs,'descend');
m.model.svbbs = m.model.svbbs(bb,:);
m.model.svxs = m.model.svxs(:,bb);
