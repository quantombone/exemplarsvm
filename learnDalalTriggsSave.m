function model = learnDalalTriggsSave(data_set, cls, params)
filer = sprintf('/csail/vision-videolabelme/databases/video_adapt/final-models/sun11-dt2-%s.mat',cls);
filerlock = [filer '.lock'];
if fileexists(filer) || mymkdir_dist(filerlock) == 0
  model = [];
  return;
end

model = learnDalalTriggs(data_set,cls);
save(filer,'model');
try
  rmdir(filerlock);
catch
  fprintf(1,'lock file already gone\n');
end
