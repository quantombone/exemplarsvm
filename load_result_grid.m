function grid = load_result_grid(models,dataset_params,setname,curthresh)
%Given a set of models, return a grid of results from those models' firings
%on the subset of images (target_directory is 'trainval' or 'test')
%[curthresh]: only keep detections above this number (-1.1 for
%esvm, .5 for vis-reg)
%Tomasz Malisiewicz (tomasz@cmu.edu)


% final_dir = ...
%     sprintf('%s/grids',dataset_params.localdir);
% if ~exist(final_dir,'dir')
%   mkdir(final_dir);
% end

% if ~exist('models','var')  
%   [cls,mode] = load_default_class;
%   models = load_all_models(cls,mode);
% end

% final_file = ...
%     sprintf('%s/grids/%s_%s_grid.mat',dataset_params.localdir,'both',models{1}.cls);

% if ~exist('curset','var')
%   curset = 'both'
% end

if ~exist('curthresh','var')
  curthresh = -1.1;
end

setname = [setname '.' models{1}.cls];

final_file = sprintf('%s/applied/%s-%s.mat',dataset_params.localdir,setname, ...
                     models{1}.models_name);

if fileexists(final_file)
  fprintf(1,'Loading final file %s\n',final_file);
  load(final_file);
  return;
end

% if ~exist('models','var')
%   error('Need to specify models');

%   models = load_all_models_chunks('train','exemplars2','100');  
%   %fprintf(1,['Need some models, cannot continue without one' ...
%   %           ' argument\n']);
%   %grid = [];
%   %return;
% end

%if ~exist('bg','var')
%fprintf(1,'Loading default set of images\n');
%[bg,setname] = default_testset;
%bg = eval(models{1}.fg);
%setname = 'sketchfg';
%curset = 'trainval'

%curset = 'both'
%curcls = models{1}.cls;
%setname = [curset '.' curcls];% models{1}.cls];
%end

% if ~exist('setname','var')
%   [bg,setname] = default_testset;
% end

% if ismember(models{1}.models_name,{'dalal'})
%   models_name = 'dalal';
% end

baser = sprintf('%s/applied/%s-%s/',dataset_params.localdir,setname, ...
                models{1}.models_name);
fprintf(1,'base directory: %s\n',baser);

files = dir([baser 'result*mat']);
grid = cell(1,length(files));

for i = 1:length(files)
  if mod(i,100) == 0
    fprintf(1,'%d/%d\n',i,length(files));
  end
  
  filer = sprintf('%s/%s', ...
                  baser,files(i).name);
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


[aa,bb] = sort(cellfun(@(x)x.index,grid));
grid = grid(bb);


lockfile = [final_file '.lock'];
if fileexists(final_file) || (mymkdir_dist(lockfile) == 0)
  return;
end

save(final_file,'grid');
if exists(lockfile,'dir')
  rmdir(lockfile);
end

%fprintf(1,'Got to end of load result grid function\n');