function m = learn_on_positive_space
%Do learning on space spanned by positives

%% load a good exemplars
res = load(['/nfs/baikal/tmalisie/local/VOC2007/exemplars/' ...
            '001221.1.train.mat']);
m = res.m;

VOCinit;
[train_set, tmp] = textread(sprintf(VOCopts.imgsetpath,'train'),['%s' ...
                    ' %d']);

[val_set, tmp] = textread(sprintf(VOCopts.imgsetpath,'val'),['%s' ...
                    ' %d']);

if sum(ismember(m.curid,train_set))
  imgset = 'train';
else
  imgset = 'val';
end


imgset = 'trainval';

m.models_name = 'iccv2001alg';
%Create initial w
m.model.startx = mean(m.model.x,2);
m.model.w = mean(m.model.x,2);
m.model.w = reshape(m.model.w, m.model.hg_size);
m.model.w = m.model.w - mean(m.model.w(:));
m.model.b = -100;
os_thresh = .5;

bg = get_pascal_bg('train',sprintf('-%s',m.cls));
mining_queue = initialize_mining_queue(bg);
myRandomize;
r = randperm(length(mining_queue));
mining_queue = mining_queue(r);
mining_params = get_default_mining_params;
mining_params.detection_threshold = -1;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 10;
mining_params.NMS_MINES_OS = 10;
mining_params.MAX_WINDOWS_BEFORE_SVM = 2000;
mining_params.MAX_IMAGES_BEFORE_SVM = 20;
mining_params.MAX_WINDOWS_PER_IMAGE = 400;
mining_params.SVMC = .0001;

%enable this
mining_params.A_FROM_POSITIVES = 1;


load xpos.mat
for iterations = 1:10

  [m, mining_queue] = learn_with_positives(m, xpos, mining_queue, bg, ...
                                           mining_params);
  
  filer = sprintf('/nfs/baikal/tmalisie/labelme400/www/newtraces/%s.%d_%05d.png',m.curid,m.objectid,iterations);
  set(gcf,'PaperPosition',[0 0 20 20]);
  print(gcf,'-dpng',filer);

  [xpos,Isv] = extract_positives(m,os_thresh,imgset);
  m.I = Isv;
end


function [xpos,Isv] = extract_positives(m,os_thresh,imgset)
%extract positives from the set of positive images

bg = get_pascal_bg(imgset,m.cls);
mining_queue = initialize_mining_queue(bg);

mining_params = get_default_mining_params;
mining_params.detection_threshold = -1000;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 10;
mining_params.NMS_MINES_OS = 0;
mining_params.MAX_WINDOWS_BEFORE_SVM = 2000000;
mining_params.MAX_WINDOWS_PER_IMAGE = 1;

[hn] = load_hn_fg(m, mining_queue, bg, mining_params);
overlaps = get_overlaps_with_gt(m, hn.objids, bg);

scores = m.model.w(:)'*hn.xs - m.model.b;
medscore = median(scores(:));

hits = find(overlaps>os_thresh & scores'>medscore);
[aa,bb] = sort(scores(hits),'descend');
hits = hits(bb);

fprintf(1,'Length hits is %d\n',length(hits));

Isv = get_sv_stack(hn.objids(hits), bg);
%figure(14)
%imagesc(Isv)
%title('current detections');
%drawnow
%pause(.1)

xpos = hn.xs(:,hits);

function [m,mining_queue] = learn_with_positives(m, xpos, mining_queue, bg, ...
                                                 mining_params)
%Learn with xpos being both the A matrix and the positives to learn
%over

m.model.x = [m.model.startx xpos];

maxiter = 4;
for k = 1:maxiter
  [m, mining_queue] = ...
      mine_negatives(m, mining_queue, bg, mining_params, k);
end
