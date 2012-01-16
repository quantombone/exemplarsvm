function files = get_file_list(dirpath)
if ~isstr(dirpath)
  fprintf(1,'get_file_list: Warning, not a directory\n');
  files = {};
  return;
end

files = dir(dirpath);
isdirs = arrayfun(@(x)x.isdir,files);
files = files(~isdirs);
files = {files.name};
files = cellfun2(@(x)[dirpath '/' x],files);
