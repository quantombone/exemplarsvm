function [g,m] = do_pedro_transfer(models)

m = models;
for i = 1:length(models)
  m{i}.models_name = 'pedroxfer';
end

addpath(genpath('/nfs/hn22/tmalisie/ddip/util/voc-release3.1/'))
load bus_final.mat
fg = get_pascal_bg('test','bus');

allbg = cat(1,get_pascal_bg('trainval'),get_pascal_bg('test'));
curids = cell(length(allbg),1);
for i = 1:length(allbg)
  [tmp,curids{i},tmp] = fileparts(allbg{i});
end

%fg = fg(1:5);
g = cell(length(fg),1);

xs = cellfun2(@(x)x.model.x,models);
xs = cat(2,xs{:});

xs2 = xs;
for i = 1:size(xs2,2)
  xs2(:,i) = xs2(:,i) - mean(xs2(:,i));
end

for i = 1:length(fg)
  I = convert_to_I(fg{i});
  boxes = detect(I,model,model.thresh);
  boxes = nms(boxes,.5);
  
  %if size(boxes,1)==0
  %  continue
  %end
  
  %newboxes are in my format, boxes is in pedros format
  newboxes = zeros(size(boxes,1),8);
  if size(boxes,1)>0
    [aa,bb] = sort(boxes(:,end),'descend');
    boxes = boxes(bb,:);
    newboxes(:,1:4) = boxes(:,1:4);
    newboxes(:,end) = boxes(:,end);
  end
  
  for j = 1:size(boxes,1)
    curm = initialize_model_dt(I,boxes(j,:),...
                               models{1}.model.params.sbin,...
                               models{1}.model.hg_size);
    
    [tmp,ind] = max(xs2'*curm.x);
    newboxes(j,6) = ind;
  end
  
  g{i}.bboxes = newboxes;
  g{i}.imbb = [1 1 size(I,2) size(I,1)];
  [tmp,g{i}.curid,tmp] = fileparts(fg{i});
  
  g{i}.index = find(ismember(g{i}.curid,curids));
  g{i}.extras = [];
  %% no extras, no coarse boxes
  
  figure(1)
  clf
  imagesc(I)
  if size(newboxes,1)>0
    plot_bbox(newboxes(1,:))
    title('Top Detection');
  end
  % figure(2)
  % [aa,bb] = sort(xs2'*curm.x,'descend');
  % for q = 1:9
  %   subplot(3,3,q)
  %   imagesc(get_exemplar_icon(models,bb(q)));
  % end
  drawnow
  
end