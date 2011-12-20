function [cls,mode] = load_default_class
%Load the default object class and object mode from disk

filer = '/nfs/baikal/tmalisie/default_class.txt';
if fileexists(filer)
  
  cls = textread(filer,'%s');
  mode = cls{2};
  cls = cls{1};
  
  % fid = fopen(filer,'r');
  % [cls] = fscanf(fid,'%s');
  % fclose(fid);

  %spacer = find(cls==' ');
  %mode = cls(spacer+1:end);
  %cls = cls(1:spacer-1);

  fprintf(1,'Loading default class from file "%s" ',filer);    
else
  fprintf(1,'No default file %s, using hardcoded class ',filer);    
  cls = 'train';
  mode = 'exemplars';
end
fprintf(1,'\n    --- cls="%s" mode="%s"\n', cls, mode);
