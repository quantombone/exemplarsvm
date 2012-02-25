function model = learnDalalTriggsSave(data_set, cls, params)
filer = sprintf('/csail/vision-videolabelme/databases/SUN11/dt-models/sun11-dt-%s.mat',cls);
filerlock = [filer '.lock'];
if fileexists(filer) || mymkdir_dist(filerlock) == 0
  model = [];
  return;
end

if ~exist('params','var')
  params = esvm_get_default_params;
end

model = learnDalalTriggs(data_set,cls,params);
save(filer,'model');
if fileexists(filerlock)
  rmdir(filerlock);
else
  fprintf(1,'lock file already gone\n');
end
