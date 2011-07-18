function dump_memex_icons(dataset_params,models)

basedir = [dataset_params.localdir '/icons/'];
if ~exist(basedir,'dir')
  mkdir(basedir);
end

for i = 1:length(models)
  fprintf(1,'.');
  mstring = sprintf('%s/%s.%d.1.png',basedir,...
                    models{i}.curid,models{i}.objectid);
  if fileexists(mstring)
    continue
  end
  
  I = get_exemplar_icon(models,i);
  ms = max(size(I,1),size(I,2));
  I = imresize(I,100/ms);
  I = max(0.0,min(1.0,I));
  I = flip_image(I);
  imwrite(I,mstring);
end