function files = get_file_list(dirpath)
% Given a directory, list all files inside the directory and create
% a cell array of strings, where each string is the file location
% NOTE: this can be used with convert_to_I, as follows
% >> files = get_file_list('~/myimages/');
% >> imagesc(convert_to_I(files{10}))

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
