function [models,mining_queue] = mine_negatives(models, mining_queue, bg, ...
                                           mining_params, iteration)
%% Mine negatives (for a set of models) and update the current
%% classifiers inside models 
%%
%% Tomasz Malisiewicz (tomasz@cmu.edu)
for q = 1:length(models)
  lastw{q} = models{q}.model.w;
  lastb{q} = models{q}.model.b;
end

%during first few iterations, we take many windows per image
if iteration <= mining_params.early_late_cutoff
  mining_params.detection_threshold = mining_params.early_detection_threshold;
else
  %in later iterations when we pass through many images, we use SVM cutoff
  mining_params.detection_threshold = mining_params.late_detection_threshold;
end

[hn, models, mining_queue, mining_stats] = ...
    load_hn_fg(models, mining_queue, bg, mining_params);

for q = 1:length(models)
  if (size(hn.xs{q},2) >= mining_params.MAX_WINDOWS_BEFORE_SVM) || ...
    (iteration == mining_params.MAXITER)
    [models{q}] = update_the_model(models,hn,q,mining_params, lastw, ...
                                   iteration, mining_stats, bg);
  end
end


function [m] = update_the_model(models,hn,index,mining_params, lastw, ...
                                iteration, mining_stats, bg)
%% UPDATE the current SVM and show the results

m = models{index};
m.iteration = m.iteration + 1;
xs = hn.xs{index};
objids = hn.objids{index};
%TODO: Remove redundant SVs here

%apply r to current iterations plot
r = m.model.w(:)'*xs-m.model.b;
rstart = r;

chosenids = 1:length(r);

%bad set is old support vectors and newly chosen objects
badx = [m.model.nsv xs(:,chosenids)];
badids = [m.model.svids objids(chosenids)];

NUM_ADDONS = sum(r>=-1.0);
fprintf(1,'got %d/%d from MINING, max=%.3f\n',...
        NUM_ADDONS,length(r),max(r));

goodx = [m.model.x];
superx = [goodx badx];

supery = cat(1,...
             +1*ones(size(goodx,2),1),...
             -1*ones(size(badx,2),1));


m3 = [];

%% if exemplar comes with a mask, then we restring learning to weights within
%% allowable region, if no mask then create a full one which
%% doesn't eliminate anything
if isfield(m.model,'mask')
  fdim = features;
  m3 = logical(repmat(m.model.mask,[1 1 fdim]));
  m3 = m3(:);
end

old_scores = m.model.w(:)'*superx - m.model.b;
[wex,b,svm_model] = do_svm(supery, superx, mining_params, m3, ...
                           m.model.hg_size, old_scores);

m.model.w = reshape(wex,m.model.hg_size);
m.model.b = b;

r = wex'*badx - b;

if strmatch(m.models_name,'dalal')
  %% here we take the best exemplars
  allscores = wex'*m.model.x - b;
  [aa,bb] = sort(allscores,'descend');
  [aabad,bbbad] = sort(r,'descend');
  maxbad = aabad(ceil(.05*length(aabad)));
  LEN = max(sum(aa>=maxbad), m.model.keepx);
  m.model.x = m.model.x(:,bb(1:LEN));
  fprintf(1,'dalal:WE NOW HAVE %d exemplars in category\n',LEN);
end

svs = find(r >= -1.0000);

%KEEP 3#SV vectors (or at least 1000 of them)
total_length = ceil(mining_params.beyond_nsv_multiplier*length(svs));
total_length = min(total_length,mining_params.max_negatives);

[alpha,beta] = sort(r,'descend');
svs = beta(1:min(length(beta),total_length));
m.model.nsv = badx(:,svs);
m.model.svids = badids(svs);

r = wex'*superx - b;

% Append new w to trace
m.model.wtrace{end+1} = m.model.w;
m.model.btrace{end+1} = m.model.b;

%if DISPLAY == 0
%  return;
%end


%% HERE WE DRAW THE FIGURES
figure(1)
clf

subplot(2,2,1)
% I = get_exemplar_icon(models,index);
% imagesc(I)
% axis image
% axis off
% cls = '';
% objid = 0;
% if isfield(m,'cls');
%   cls = m.cls;
% end
% if isfield(m,'objectid')
%   objid = m.objectid;
% end
% title(sprintf('Exemplar %s %s.%d',cls,m.curid,objid),...
%       'interpreter','none')

% mx = mean(m.model.x,2);
% raw_picture = HOGpicture(reshape(mx-mean(mx(:)),m.model.hg_size));
% pos_picture = HOGpicture(m.model.w);
% neg_picture = HOGpicture(-m.model.w);
% spatial_picture = sum(m.model.w.*reshape(mean(m.model.x,2), ...
%                                          size(m.model.w)),3);
% spatial_picture = imresize(spatial_picture,[size(pos_picture,1) ...
%                     size(pos_picture,2)],'nearest');

% raw_picture = raw_picture - min(raw_picture(:));
% raw_picture = raw_picture / max(raw_picture(:));

% pos_picture = pos_picture - min(pos_picture(:));
% pos_picture = pos_picture / max(pos_picture(:));

% neg_picture = neg_picture - min(neg_picture(:));
% neg_picture = neg_picture / max(neg_picture(:));

% spatial_picture = spatial_picture - min(spatial_picture(:));
% spatial_picture = spatial_picture / max(spatial_picture(:));

% pos_picture = pad_image(pos_picture,10);
% neg_picture = pad_image(neg_picture,10);
% raw_picture = pad_image(raw_picture,10);
% spatial_picture = pad_image(spatial_picture,10);

% res_picture = cat(1,cat(2,pos_picture,neg_picture),...
%                   cat(2,raw_picture,spatial_picture));

% subplot(2,2,2)
% imagesc(res_picture);
% axis image
% axis off
% template_diff = norm(m.model.w(:)-lastw{index}(:));
% title(sprintf('(+w,-w,+nhog,spatial)\ndiff from last: %.5f',template_diff))

subplot(1,2,1)
LENX = sum(supery==-1);
plot([1 LENX],[-1 -1],'g--','LineWidth',2)
hold on;
plot([1 LENX],[1 1],'g--','LineWidth',2)
hold on;
plot([1 LENX],[0 0],'k','LineWidth',2)
hold on;
plot(linspace(1,LENX,sum(supery==1)),r(supery==1),'r.');
hold on;
plot(1:LENX,old_scores(supery==-1),'m.','MarkerSize',2);
hold on;
plot(1:LENX,r(supery==-1),'b.');
axis([1 LENX min(r) max(1.05,max(r))]);
mr = max(rstart);
title(sprintf(['Iter %d\n MaxMine = %.3f, #ViolI=%d #EmptyI=%d'],...
              m.iteration,mr,mining_stats.num_violating,...
              mining_stats.num_empty));

xlabel('id index')
ylabel('SVM score');

subplot(1,2,2)
Isv = get_sv_stack(m.model.svids(1:min(length(m.model.svids),25)),bg,m,5,5);
imagesc(Isv)
axis image
axis off
title('Top 25 -SVs');
drawnow

if (mining_params.dump_images == 1) || ...
      (mining_params.dump_last_image == 1 && ...
       m.iteration == mining_params.MAXITER)
  set(gcf,'PaperPosition',[0 0 20 10]);
  print(gcf,sprintf('%s/%s.%d_iter=%05d.png', ...
                    mining_params.final_directory,m.curid,...
                    m.objectid,m.iteration),'-dpng');
 
end
 
