function cls = load_default_class
%Load the default object class from disk

filer = '/nfs/baikal/tmalisie/default_class.txt';
if fileexists(filer)
  fid = fopen(filer,'r');
  cls = fscanf(fid,'%s');
  fclose(fid);
  fprintf(1,'Loading default class from file "%s" ',filer);    
else
  fprintf(1,'No default file %s, using hardcoded class ',filer);    
  cls = 'train';
end
fprintf(1,'cls="%s"\n',cls);
