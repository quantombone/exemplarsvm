function do_show_stacks(models)
%% train all exemplars

VOCinit;

if ~exist('models','var')
  models = load_all_models;
end

mode = models{1}.models_name;
results_directory = ...
    sprintf('%s/%s/',VOCopts.wwwdir,mode);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

myRandomize;
rrr = randperm(length(models));

bg = get_pascal_bg('trainval');
for ix = 1:length(models)
  i = rrr(ix);
  filer = sprintf('%s/%s.%d.%s.png',results_directory,...
                  models{i}.curid,...
                  models{i}.objectid,...
                  models{i}.cls);
  
  filerlock = [filer '.lock'];
  if fileexists(filer) || (mymkdir_dist(filerlock)==0)
    continue
  end

  Isv = get_sv_stack(models{i},bg,10,10);
  imwrite(Isv,filer);
  rmdir(filerlock);

end
