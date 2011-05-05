function save_default_class(cls)
%Save the default class to disk

fid = fopen('/nfs/baikal/tmalisie/default_class.txt','w');
fprintf(fid,cls);
fclose(fid);
