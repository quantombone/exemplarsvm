function dump_memex_icons(dataset_params,models)

basedir = [dataset_params.localdir '/icons/'];
if ~exist(basedir,'dir')
  mkdir(basedir);
end

MAXDIM = 200;
for i = 1:length(models)
  fprintf(1,'.');
  mstring0 = sprintf('%s/%s.%d.0.png',basedir,...
                    models{i}.curid,models{i}.objectid);
  mstring1 = sprintf('%s/%s.%d.1.png',basedir,...
                    models{i}.curid,models{i}.objectid);
  %if fileexists(mstring)
  %  continue
  %end
  
  I = get_exemplar_icon(models,i);
  ms = max(size(I,1),size(I,2));
  I = imresize(I,MAXDIM/ms);
  I = max(0.0,min(1.0,I));
  imwrite(I,mstring0);
  I = flip_image(I);
  imwrite(I,mstring1);
end