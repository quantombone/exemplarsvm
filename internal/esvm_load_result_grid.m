function grid = esvm_load_result_grid(dataset_params,...
                                      models,setname,files,curthresh)
%Given a set of models, return a grid of results from those models' firings
%on the subset of images (target_directory is 'trainval' or 'test')
%[curthresh]: only keep detections above this number (-1.1 for
%esvm, .5 for vis-reg)
%Tomasz Malisiewicz (tomasz@cmu.edu)

fullsetname = [setname];

final_file = sprintf('%s/detections/%s-%s.mat',...
                     dataset_params.localdir,fullsetname, ...
                     models{1}.models_name);

if fileexists(final_file)
  res = load(final_file);
  grid = res.grid;
  return;
end

wait_until_all_present(files,5);

if ~exist('curthresh','var')
  curthresh = -1.1;
end


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

baser = sprintf('%s/detections/%s-%s/',dataset_params.localdir,setname, ...
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
    
      %curids = 1:1250;
      %iscurid = (ismember(grid{i}.res{j}.bboxes(:,6),curids));
      
      
      goods = find(grid{i}.res{j}.bboxes(:,end) >= curthresh);% & iscurid);
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

if 1
  
  %BUG: I'm not sure that pruning here isn't going to hurt me later
  %since we no longer have a direct mapping between detections and the
  %test-set ids
  
  %Prune away files which didn't load
  lens = cellfun(@(x)length(x),grid);
  grid = grid(lens>0);
  grid = cellfun2(@(x)x.res,grid);
  grid2 = grid;
  grid = [grid2{:}];
else
  fprintf(1,'warning skipping pruning\n');
end

if length(grid) > 0
  
  %sort grids by image index
  [aa,bb] = sort(cellfun(@(x)x.index,grid));
  grid = grid(bb);
  
  %only keep grids with at least one detection
  %lens = cellfun(@(x)size(x.bboxes,1),grid);
  %grid = grid(lens>0);
end

%if fileexists(final_file) || (mymkdir_dist(lockfile) == 0)
%  wait_until_all_present({final_file},5);
%  return;
%else
save(final_file,'grid');

if exist(lockfile,'dir')
  rmdir(lockfile);
end

f = dir(final_file);
if (f.bytes < 1000)
  fprintf(1,'warning file too small, not saved\n');
  delete(final_file);
end

if fileexists(final_file)
  %Clean up individual files which were written to disk
  for i = 1:length(files)
    delete(files{i});
  end
  %Delete directory too
  [basedir,other,ext] = fileparts(files{1});
  rmdir(basedir);
end