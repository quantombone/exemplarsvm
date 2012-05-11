function savemodel(model,cls)

save(sprintf('/csail/vision-videolabelme/databases/SUN11/final-models/longmine-%s.mat',cls),'model');
%try
  save(sprintf('~/home_ac_a-5-%s.mat',cls),'model');
catch
  fprintf(1,'Could not save model\n');
end