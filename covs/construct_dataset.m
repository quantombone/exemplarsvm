%addpath(genpath('~/exemplarsvm'));

image_set = textread('images','%s');
newfiles = cellfun(@(x)strrep(strrep(x,'/Images', ...
                                     '/Annotations'),...
                              '.jpg','.xml'),image_set, ...
                   'UniformOutput',false);

data_set = cell(length(image_set),1);
parfor i = 1:length(image_set)
  [annotation, xml] = loadXML(newfiles{i});
  annotation = annotation.annotation;
  boundingbox = LMobjectboundingbox(annotation);
  annotation.objects = annotation.object;
  annotation = rmfield(annotation,'object');
fprintf(1,'.');
  for m = 1:length(annotation.objects)
    annotation.objects(m).class = annotation.objects(m).name;
    annotation.objects(m).truncated = 0;%annotation.objects(m).crop;
    annotation.objects(m).difficult = 0;%annotation.objects(m).crop;
    annotation.objects(m).bndbox.xmin = boundingbox(m,1);
    annotation.objects(m).bndbox.ymin = boundingbox(m,2);
    annotation.objects(m).bndbox.xmax = boundingbox(m,3);
    annotation.objects(m).bndbox.ymax = boundingbox(m,4);
    annotation.objects(m).bbox = boundingbox(m,1:4);
  end

  data_set{i} = annotation;
  data_set{i}.I = image_set{i};
end
