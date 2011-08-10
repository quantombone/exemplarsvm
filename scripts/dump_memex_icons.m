function dump_memex_icons(dataset_params,models)
%Dump out little image icons (plus little hog image icons)

basedir = [dataset_params.localdir '/icons/'];
if ~exist(basedir,'dir')
  mkdir(basedir);
end

basedir2 = [dataset_params.localdir '/icons/' models{1}.cls];
if ~exist(basedir2,'dir')
  mkdir(basedir2);
end

MAXDIM = 200;
for i = 1:length(models)
  fprintf(1,'.');
  mstring0 = sprintf('%s/%s.%d.0.png',basedir,...
                    models{i}.curid,models{i}.objectid);
  mstring1 = sprintf('%s/%s.%d.1.png',basedir,...
                    models{i}.curid,models{i}.objectid);
  
  mstring0 = sprintf('%s/%s/%s.%s.%d.0.png',basedir,...
                    models{i}.cls,models{i}.cls,models{i}.curid,models{i}.objectid);
  mstring1 = sprintf('%s/%s/%s.%s.%d.1.png',basedir,...
                    models{i}.cls,models{i}.cls,models{i}.curid,models{i}.objectid);
  
  hstring0 = sprintf('%s/%s/%s.hog.%s.%d.0.png',basedir,...
                    models{i}.cls,models{i}.cls,models{i}.curid,models{i}.objectid);
  hstring1 = sprintf('%s/%s/%s.hog.%s.%d.1.png',basedir,...
                    models{i}.cls,models{i}.cls,models{i}.curid,models{i}.objectid);
  
  %if fileexists(mstring0) && fileexists(mstring1)
  %  continue
  %end
  
  I = get_exemplar_icon(models,i);
  ms = max(size(I,1),size(I,2));
  I = imresize(I,MAXDIM/ms);
  I = max(0.0,min(1.0,I));
  imwrite(I,mstring0);
  I = flip_image(I);
  imwrite(I,mstring1);
  
  I2 = HOGpicture(models{i}.model.w);
  I2 = jettify(I2);
  imwrite(I2,hstring0);
  I2 = flip_image(I2);
  imwrite(I2,hstring1);
end