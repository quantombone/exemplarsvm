function grid = load_result_grid(models,curset)
%Given a set of models, return a grid of results from those models' firings
%on the subset of images (target_directory is 'trainval' or 'test')
%if only_images_of_class is specified, then only the subset of
%images containing elements of that class are loaded

%Tomasz Malisiewicz (tomasz@cmu.edu)
VOCinit;


% final_dir = ...
%     sprintf('%s/grids',VOCopts.localdir);
% if ~exist(final_dir,'dir')
%   mkdir(final_dir);
% end

% if ~exist('models','var')  
%   [cls,mode] = load_default_class;
%   models = load_all_models(cls,mode);
% end

% final_file = ...
%     sprintf('%s/grids/%s_%s_grid.mat',VOCopts.localdir,'both',models{1}.cls);

if ~exist('curset','var')
  curset = 'both'
end
%curset = 'trainval';
curcls = models{1}.cls;
setname = [curset '.' curcls];

final_file = sprintf('%s/applied/%s-%s.mat',VOCopts.localdir,setname, ...
                     models{1}.models_name);



% if fileexists(final_file)
%   fprintf(1,'Loading final file %s\n',final_file);
%   load(final_file);
%   return;
% end


% if ~exist('models','var')
%   error('Need to specify models');

%   models = load_all_models_chunks('train','exemplars2','100');  
%   %fprintf(1,['Need some models, cannot continue without one' ...
%   %           ' argument\n']);
%   %grid = [];
%   %return;
% end

%if ~exist('bg','var')
  fprintf(1,'Loading default set of images\n');
  %[bg,setname] = default_testset;
  %bg = eval(models{1}.fg);
  %setname = 'sketchfg';
  %curset = 'trainval'
 
  %curset = 'both'
  curcls = models{1}.cls;
  setname = [curset '.' curcls];% models{1}.cls];
%end


% if ~exist('setname','var')
%   [bg,setname] = default_testset;
% end



% if ismember(models{1}.models_name,{'dalal'})
%   models_name = 'dalal';
% end

baser = sprintf('%s/applied/%s-%s/',VOCopts.localdir,setname, ...
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
    
    newid = str2num(grid{i}.res{j}.curid);
    grid{i}.res{j}.bboxes(:,11) = newid;
    grid{i}.res{j}.coarse_boxes(:,11) = newid;
    goods = (grid{i}.res{j}.bboxes(:,end) >= -1.1);

    grid{i}.res{j}.bboxes = grid{i}.res{j}.bboxes(goods,:);
    grid{i}.res{j}.coarse_boxes = ...
        grid{i}.res{j}.coarse_boxes(goods,:);
    grid{i}.res{j}.extras.maxos = ...
        grid{i}.res{j}.extras.maxos(goods);

    grid{i}.res{j}.extras.maxind = ...
        grid{i}.res{j}.extras.maxind(goods);
    
    grid{i}.res{j}.extras.maxclass = ...
        grid{i}.res{j}.extras.maxclass(goods);

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

%fprintf(1,'Loaded grids, saving to %s\n',final_file);
%save(final_file,'grid');

fprintf(1,'Got to end of load result grid function\n');