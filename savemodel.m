function savemodel(model,cls)

save(sprintf('/csail/vision-videolabelme/databases/SUN11/dt-models/longmine.%s.mat',cls),'model');
