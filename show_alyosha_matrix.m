function show_alyosha_matrix(models)

%%% Mine some negative windows
bg = get_pascal_bg('trainval',['-' models{1}.cls]);

localizeparams.thresh = -100.0;
localizeparams.TOPK = 10;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;
localizeparams.FLIP_LR = 1;
localizeparams.NMS_MINES_OS = 0.5;
NIM = 0;
allx = [];
allids = [];

for q = 1:NIM
  I = convert_to_I(bg{q});
  [tmp,curid,tmp] = fileparts(bg{q});
  r = randperm(length(models));
  r = r(1);
  [rs,t] = localizemeHOG(I,models(r),localizeparams);
  ids = [rs.id_grid{:}];
  xs = [rs.support_grid{:}];
  xs = cat(2,xs{:});
  for i = 1:length(ids)
    ids{i}.curid = curid;
  end
  allx = cat(2,allx,xs);
  allids = cat(2,allids,ids);
end

for i = 1:length(allids)
  clear m;
  m.curid = allids{i}.curid;
  m.gt_box = allids{i}.bb;
  m.models_name = 'fake';
  m.model.x = allx(:,i);
  models{end+1} = m;
end

xs = cellfun2(@(x)x.model.x,models);
xs = cat(2,xs{:});

xs2 = xs;
for i = 1:size(xs2,2)
  xs2(:,i) = xs2(:,i) - mean(xs2(:,i));
end

xs3 = xs;
for i = 1:size(xs3,2)
  xs3(:,i) = xs3(:,i) / norm(xs3(:,i));
end

%d = -distSqr_fast(xs,xs);
%d = xs3'*xs3;
d = xs2'*xs;
%d = xs'*xs2;
%d = xs2'*xs2;

for i = 1:size(d,1)
  [aa,bb] = sort(d(i,:),'descend');
  scores(i) = aa(2);
end
%[aa,r] = sort(scores,'descend');

%r = randperm(length(models));
r = 1:length(models);
for iii = 1:length(r)
  index = r(iii);

% figure(1)
% [aa,bb] = sort(d(:,index));
% show_sorting(models,aa,bb);
% title('raw NN')

% d = xs'*xs;
% figure(2)
% [aa,bb] = sort(d(:,index),'descend');
% show_sorting(models,aa,bb);
% title('dot product')

% d = xs2'*xs;
% figure(3)
% [aa,bb1] = sort(d(:,index),'descend');
% show_sorting(models,aa,bb1);
% title('dot product left-normalize')

%d = xs2'*xs2;
figure(4)

[aa,bb2] = sort(d(:,index),'descend');
show_sorting(models,aa,bb2);
title('dot product right-normalize')

% d = xs3'*xs3;
% figure(4)
% [aa,bb] = sort(d(:,index),'descend');
% show_sorting(models,aa,bb);
% title('dot product l2 normalized')
pause
end
function show_sorting(models,aa,bb)
for i = 1:9
  subplot(3,3,i)
  imagesc(get_exemplar_icon(models,bb(i)));
  title(num2str(aa(i)))
end
