function draw_memex(dataset_params, models, val_grid, val_set, M)
%% draw the memex with connections that have calibrated score above
%.5

%% take all outgoing edges for each exemplar that calibrate to
%above .5

clsid = find(ismember(dataset_params.classes,models{1}.cls))

N = length(models)*2;
A = zeros(N,N);

exstrings = cell(N,1);
exstrings2 = cell(N,1);
for i = 1:length(models)
  exstrings{i} = sprintf('%s.%d.0.png',models{i}.curid, ...
                         models{i}.objectid);
  
  exstrings{length(models)+i} = sprintf('%s.%d.1.png',models{i}.curid, ...
                         models{i}.objectid);
  
  
  exstrings2{i} = sprintf('%s.%d.1.png',models{i}.curid, ...
                         models{i}.objectid);
  
  exstrings2{length(models)+i} = sprintf('%s.%d.0.png',models{i}.curid, ...
                         models{i}.objectid);
end

%hitlist = {};

for i = 1:length(val_grid)  
  bbs = val_grid{i}.bboxes;
  bbs = calibrate_boxes(bbs,M.betas);
  os = val_grid{i}.extras.maxos;
  classes = val_grid{i}.extras.maxclass;
  ids = val_grid{i}.extras.maxind;
  %mb = max(bbs(:,end));
  
  goods = find(bbs(:,end)>.9 & os > .5 & classes == clsid);
  if length(goods) == 0
    continue
  end
  
  hitstrings = cell(length(goods),1);
  hitstrings2 = cell(length(goods),1);
  for j = 1:length(goods)
    isflip = double(bbs(goods(j),7)==1);
    hitstrings{j} = sprintf('%s.%d.%d.png',val_grid{i}.curid,...
                            ids(goods(j)),isflip);
    
    hitstrings2{j} = sprintf('%s.%d.%d.png',val_grid{i}.curid,...
                            ids(goods(j)),1-isflip);
  end
  
  %no flips
  %flip0 = goods(bbs(goods,7)==0);
  
  %yes flips
  %flip1 = goods(bbs(goods,7)==1);
  
  %curids = ids(flip0);
  
  [tmp1,target_ids] = ismember(hitstrings,exstrings);
  source_strings = exstrings(bbs(goods,6));
  [tmp2,source_ids] = ismember(source_strings,exstrings);
  scores = bbs(goods,end);
  
  [tmp12,target_ids2] = ismember(hitstrings2,exstrings);
  source_strings2 = exstrings(bbs(goods,6));
  [tmp22,source_ids2] = ismember(source_strings2,exstrings2);
  %scores = bbs(goods,end);
    
  goods2 = find(tmp1 & tmp2);
  target_ids = target_ids(goods2);
  source_ids = source_ids(goods2);
  
  target_ids2 = target_ids2(goods2);
  source_ids2 = source_ids2(goods2);
  scores = scores(goods2);
  
  for j = 1:length(target_ids)
    A(source_ids(j),target_ids(j)) = max(A(source_ids(j),target_ids(j)),...
                                         scores(j));
    
    A(source_ids2(j),target_ids2(j)) = max(A(source_ids2(j),target_ids2(j)),...
                                         scores(j));
  end
  
  fprintf(1,'.');
end

A = (A + A')/2;
A = A - diag(diag(A));

keepers = find((sum(A,1)>0));
A2 = A(keepers,keepers);


other.icon_string = @(i)sprintf('image="%s"', ...
                                exstrings{keepers(i)});

other.svg_file = sprintf('%s/icons/%s.pdf', ...
                  dataset_params.localdir, models{1}.cls);
other.gv_file = sprintf('%s/icons/%s.gv', ...
                  dataset_params.localdir, models{1}.cls);

%other.svg_file = svgfile;
make_memex_graph(A2,other);


