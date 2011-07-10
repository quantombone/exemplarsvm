function grid = load_result_grid(dataset_params,models,setname,files,curthresh)
%Given a set of models, return a grid of results from those models' firings
%on the subset of images (target_directory is 'trainval' or 'test')
%[curthresh]: only keep detections above this number (-1.1 for
%esvm, .5 for vis-reg)
%Tomasz Malisiewicz (tomasz@cmu.edu)

wait_until_all_present(files,5);

if ~exist('curthresh','var')
  curthresh = -1.1;
end

setname = [setname '.' models{1}.cls];

final_file = sprintf('%s/applied/%s-%s.mat',dataset_params.localdir,setname, ...
                     models{1}.models_name);
lockfile = [final_file '.lock'];

if fileexists(final_file) || (mymkdir_dist(lockfile)==0)
  %wait until lockfile is gone
  wait_until_all_present({lockfile},5,1);
  fprintf(1,'Loading final file %s\n',final_file);
  res = load_keep_trying(final_file,5);
  grid = res.grid;
  return;
end

%if we got here, then the final file isn't there, and we were able
%to write a lock file successfully

baser = sprintf('%s/applied/%s-%s/',dataset_params.localdir,setname, ...
                models{1}.models_name);
fprintf(1,'base directory: %s\n',baser);

%with the dir command partial results could be loaded 
%files = dir([baser 'result*mat']);
grid = cell(1,length(files));

for i = 1:length(files)
  if mod(i,100) == 0
    fprintf(1,'%d/%d\n',i,length(files));
  end
  
  %filer = sprintf('%s/%s', ...
  %                baser,files(i).name);
  filer = files{i};
  stuff = load(filer);
  grid{i} = stuff;
  
  for j = 1:length(grid{i}.res)
    
    index = grid{i}.res{j}.index;
    
    if size(grid{i}.res{j}.bboxes,1) > 0
      grid{i}.res{j}.bboxes(:,11) = index;
      grid{i}.res{j}.coarse_boxes(:,11) = index;
    
      goods = find(grid{i}.res{j}.bboxes(:,end) >= curthresh);
      grid{i}.res{j}.bboxes = grid{i}.res{j}.bboxes(goods,:);
      grid{i}.res{j}.coarse_boxes = ...
          grid{i}.res{j}.coarse_boxes(goods,:);
    
      if ~isempty(grid{i}.res{j}.extras)
        grid{i}.res{j}.extras.maxos = ...
            grid{i}.res{j}.extras.maxos(goods);
        
        grid{i}.res{j}.extras.maxind = ...
            grid{i}.res{j}.extras.maxind(goods);
        
        grid{i}.res{j}.extras.maxclass = ...
            grid{i}.res{j}.extras.maxclass(goods);
      end
    end
  end
end

%Prune away files which didn't load
lens = cellfun(@(x)length(x),grid);
grid = grid(lens>0);
grid = cellfun2(@(x)x.res,grid);
grid2 = grid;
grid = [grid2{:}];

if length(grid) > 0
  
  %sort grids by image index
  [aa,bb] = sort(cellfun(@(x)x.index,grid));
  grid = grid(bb);
  
  %only keep grids with at least one detection
  lens = cellfun(@(x)size(x.bboxes,1),grid);
  grid = grid(lens>0);
end

%if fileexists(final_file) || (mymkdir_dist(lockfile) == 0)
%  wait_until_all_present({final_file},5);
%  return;
%else
save(final_file,'grid');
if exist(lockfile,'dir')
  rmdir(lockfile);
end