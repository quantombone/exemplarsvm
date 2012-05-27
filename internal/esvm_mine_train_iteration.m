function [m,mining_queue] = esvm_mine_train_iteration(m, mining_queue)
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
% if ~isfield(m.models{1}, 'wtrace')
%   m.models{1}.wtrace{1} = m.models{1}.w;
%   m.models{1}.btrace{1} = m.models{1}.b;
% end

if length(mining_queue) == 0
  fprintf(1,' ---Null mining queue, not mining!\n');
  return;
end

[hn, mining_queue, mining_stats, m] = ...
    esvm_mine_negatives(m, mining_queue);



m.models{1} = add_new_detections(m.models{1}, ...
                                 hn.xs{1}, ...
                                 hn.bbs{1});
   
m.models{1} = update_the_model(m.models{1}, mining_stats);

if m.params.display == 1
  show_figures(m);
end

function [m] = update_the_model(m, mining_stats)
%% UPDATE the current SVM, keep max number of svs, and show the results

if ~isfield(m,'mining_stats')
  m.mining_stats{1} = mining_stats;
else
  m.mining_stats{end+1} = mining_stats;
end

%m = esvm_update_positives(m,1);
%m = esvm_update_positives(m,0);
m = m.params.training_function(m);

% Append new w to trace
% m.wtrace{end+1} = m.w;
% m.btrace{end+1} = m.b;

% if (m.params.dfun == 1)
%   r = m.model.w(:)'*bsxfun(@minus,m.model.svxs,m.model.x(:,1)).^2 - ...
%       m.model.b;
% else
%   r = m.model.w(:)'*m.model.svxs - m.model.b;
% end
% m.model.svbbs(:,end) = r;

function show_figures(m)

if m.params.display == 1    
  figure(99)
  clf
  subplot(2,1,1)
  show_model_data(m, 10);
  axis off
  subplot(2,1,2)
  rpos = m.models{1}.w(:)'*m.models{1}.x - m.models{1}.b;
  rneg = m.models{1}.w(:)'*m.models{1}.svxs - m.models{1}.b;
  rneg = rneg(rneg>=-1.01);
  maxlen = max(length(rpos),length(rneg));
  xpos = round(linspace(1,maxlen,length(rpos)));
  xneg = round(linspace(1,maxlen,length(rneg)));
  topper = max([max(rpos) max(rneg)]);
  plot(xpos,rpos,'g.')
  hold on;
  plot(xneg,rneg,'k.')
  hold on;
  plot([1 maxlen],[-1 -1],'b-')
  hold on;
  plot([1 maxlen],[0 0],'k--')
  hold on;
  plot([1 maxlen],[1 1],'r--')
  axis([1 maxlen -1.1 topper+.01])
  drawnow
  snapnow
  

  
  % poshist = hist(rpos,linspace(-1,max(rpos)+.1,50));
  % neghist = hist(rneg,linspace(-1,max(rpos)+.1,50));
  
  % poshist = conv(poshist,fspecial('gaussian',[1 7]),'same');
  % neghist = conv(neghist,fspecial('gaussian',[1 7]),'same');
  % poshist = poshist/sum(poshist);
  % neghist = neghist/sum(neghist);
  
  % y2 = linspace(-1,max(rpos)+.1,50);
  % plot(poshist*.1*maxlen,y2,'r')
  % hold on
  % plot(neghist*.1*maxlen,y2,'b')
end


% if there is a local directory, and dump images was enabled, or
% dump_last_image and we are on the last iteration
if (length(m.params.localdir)>0) && ...
      ((m.params.dump_images == 1))
  
  if ((m.params.dump_last_image == 1) && ...
      (m.iteration ~= m.params.train_max_mine_iterations))
    return;
  end

  Isv1 = show_model_data(m, 15);
  imwrite(Isv1,sprintf('%s/models/%s.%d_mineiter_I=%05d.png', ...
                    m.params.localdir, m.model_name, ...
                    m.models{1}.identifier, m.iteration), 'png');
end

%old code: also show the SVM scores
% figure(1)
% clf
% show_cool_os(m)

% if (params.dump_images == 1) || ...
%       (params.dump_last_image == 1 && ...
%        m.iteration == params.train_max_mine_iterations)
%   set(gcf,'PaperPosition',[0 0 10 3]);
%   print(gcf,sprintf('%s/%s.%d_iter=%05d.png', ...
%                     params.final_directory,m.curid,...
%                     m.objectid,m.iteration),'-dpng'); 
% end

function m = add_new_detections(m, xs, bbs)
% Add current detections (xs,bbs) to the model struct (m)
% making sure we prune away duplicates, and then sort by score
%

%First iteration might not have support vector information stored
%if ~iasfield(m, 'svxs') || isempty(m.svxs)
%  m.svxs = [];
%  m.svbbs = [];
%end

if numel(bbs) == 0
  bbs = zeros(0,12);
end

m.svxs(:,end+1:end+size(xs,2)) = xs;
m.svbbs(end+1:end+size(bbs,1),:) = bbs;
%m.svxs = cat(2, m.svxs, xs);
%m.svbbs = cat(1, m.svbbs, bbs);

% %Create a unique string identifier for each of the support vectors,
% %so we can run 'unique' on them
% names = cell(size(m.svbbs, 1), 1);
% for i = 1:length(names)
%   bb = m.svbbs(i,:);
%   names{i} = sprintf('%d.%.3f.%d.%d.%d',bb(11),bb(8), ...
%                              bb(9),bb(10),bb(7));
% end
% [unames,subset,j] = unique(names);

[~,subset,~] = unique(m.svbbs(:,[1:4 7 11]),'rows');
%if norm(sort(subset)-sort(subset2))
%  keyboard
%end
bads = setdiff(1:size(m.svbbs,1),subset);
m.svbbs(bads,:) = [];
m.svxs(:,bads) = [];
%m.svbbs = m.svbbs(subset,:);
%m.svxs = m.svxs(:,subset);

% if numel(m.svxs) > 0
%   [aa,bb] = sort(m.w(:)'*m.svxs,'descend');
%   m.svbbs = m.svbbs(bb,:);
%   m.svxs = m.svxs(:,bb);
% end
