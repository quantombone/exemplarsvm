function [bboxes,maxos,maxind,maxclass] = prune_grid(models,grid)

bboxes = cell(length(models),1);
maxos = cell(length(models),1);
maxind = cell(length(models),1);
maxclass = cell(length(models),1);

for i = 1:length(models)
  fprintf(1,'.');
  curbbs = cellfun2(@(x)x.bboxes(x.bboxes(:,6)==i,:), grid);
  curbbs = cat(1,curbbs{:});
  
  curos = cellfun2(@(x)x.extras.maxos(x.bboxes(:,6)==i), grid);
  curos = cat(1,curos{:});

  curind = cellfun2(@(x)x.extras.maxind(x.bboxes(:,6)==i), grid);
  curind = cat(1,curind{:});

  curclass = cellfun2(@(x)reshape(x.extras.maxclass(x.bboxes(:,6)==i),[],1), ...
                      grid);

  curclass = cat(1,curclass{:});

    
  [aa,bb] = sort(curbbs(:,end),'descend');
  TOP = 1000;
  bb = bb(1:min(length(bb),TOP));
  bboxes{i} = curbbs(bb,:);
  maxos{i} = curos(bb);
  maxind{i} = curind(bb);
  maxclass{i} = curclass(bb);
  

end

bboxes = cat(1,bboxes{:});
maxos = cat(1,maxos);
maxind = cat(1,maxind);
maxclass = cat(1,maxclass);