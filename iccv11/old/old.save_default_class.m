function save_default_class(cls,mode)
%Save the default class and mode to disk

fid = fopen('/nfs/baikal/tmalisie/default_class.txt','w');
fprintf(fid,cls);
fprintf(fid,' ');
fprintf(fid,mode);
fclose(fid);
