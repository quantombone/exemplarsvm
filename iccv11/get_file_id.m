function id = get_file_id(filer)
[tmp,curid,tmp] = fileparts(filer);
id = str2num(curid);