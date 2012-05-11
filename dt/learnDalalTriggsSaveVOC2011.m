function model = learnDalalTriggsSaveVOC2011(data_set, cls, params)
filer = sprintf('/csail/vision-videolabelme/databases/SUN11/dt-models/voc2011-dt-%s.mat',cls);
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
try
  rmdir(filerlock);
catch
  fprintf(1,'lock file already gone\n');
end

