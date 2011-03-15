function [newmodels] = ...
    connect_scores(models, index)
%Get neighbor scores across all instances

newmodels{1} = models{index};

m = models{index};
bg = get_pascal_bg('trainval',m.cls);
mining_queue = initialize_mining_queue(bg);


mining_params = get_default_mining_params;
mining_params.detection_threshold = -1.0;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 10;
mining_params.NMS_MINES_OS = .5;
mining_params.MAX_WINDOWS_BEFORE_SVM = 2000000;
mining_params.MAX_WINDOWS_PER_IMAGE = 100;

[hn] = load_hn_fg(m, mining_queue, bg, mining_params);

%Get scores, sort by them
scores = m.model.w(:)'*hn.xs - m.model.b;
[aa,bb] = sort(scores, 'descend');

Isv = get_sv_stack(hn.objids(bb), bg);

figure(1)
clf
imagesc(Isv)
title('original hits');

return
overlaps = [hn.objids.maxos];

keyboard

[aa,bb] = sort(overlaps,'descend');

goods = find(aa>=.5);
bads = find(aa<.2);

superx = [m.model.x hn.xs(:,bb(goods)) hn.xs(:,bb(bads))];
supery = [ones(1,size(m.model.x,2)) ones(1,length(goods)) -1*ones(1,length(bads))];

%diffx = -diff(hn.xs(:,bb(goods))')';
%superx = [superx diffx];
%supery = [supery ones(1,size(diffx,2))];

supery = supery';


A = get_dominant_basis(reshape(mean(m.model.x,2), ...
                               m.model.hg_size));
newx = A'*superx;
model = liblinear_train(supery, sparse(newx)', sprintf('-B 1 -s 3 -c %f',.01));
wex = model.w(1:end-1)';
b = -model.w(end);

wex = A*wex;
%b = 0;

newmodels{2} = models{index};
newmodels{2}.objectid = newmodels{2}.objectid*-1;
newmodels{2}.model.w = reshape(wex,newmodels{2}.model.hg_size);
newmodels{2}.model.b = 0;
r = wex'*hn.xs;

figure(2)
plot(overlaps,r,'r.');
title('overlaps')

figure(3)
[aa,bb] = sort(r,'descend');

Isv = get_sv_stack(hn.objids(bb), bg);
imagesc(Isv)
title('sorted by new ranking');

figure(4)
wex = reshape(wex,m.model.hg_size);
subplot(1,2,1)
imagesc(HOGpicture(m.model.w))
title('original w')
subplot(1,2,2)
imagesc(HOGpicture(wex))
title('new w')



% function overlaps = get_overlaps(m, objids, bg)
% overlaps = zeros(length(objids),1);

% %load each image at once
% uids = unique([objids.curid]);
% VOCinit;

% for i = 1:length(uids)
%   [tmp,curid] = fileparts(bg{uids(i)});
%   recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
%   gt_bb = cat(1,recs.objects.bbox);
%   hits = find(ismember({recs.objects.class},m.cls));
%   if length(hits) == 0
%     continue
%   end
%   gt_bb = gt_bb(hits,:);
  
%   hits = find([objids.curid]==uids(i));
%   startbb = cat(1,objids(hits).bb);
%   endbb = startbb*0;
%   for j = 1:size(startbb,1)
%     xform = find_xform(m.model.coarse_box, startbb(j,:));
%     endbb(j,:) = apply_xform(m.gt_box,xform);
%   end
%   osmat = getosmatrix_bb(endbb,gt_bb);
%   maxos = max(osmat,[],2);
%   overlaps(hits) = maxos;
% end
