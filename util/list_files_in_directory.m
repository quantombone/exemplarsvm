function files = list_files_in_directory(dirpath, extensions)
% Given a directory, list all files inside the directory and create a cell
% array of strings, where each string is the file location
%
% extensions is optional and a cell array of extensions that you want this
% function to return. If given, any file with an extension not listed in this
% cell array will be not be returned. If extensions is true, then only images
% will be returned.
%
% NOTE: this can be used with convert_to_I, as follows >> files =
% list_files_in_directory('~/myimages/'); >> imagesc(convert_to_I(files{10}))

if ~isstr(dirpath)
  fprintf(1,'list_files_in_directory: Warning, not a directory\n');
  files = {};
  return;
end

files = dir(dirpath);
isdirs = arrayfun(@(x)x.isdir,files);
files = files(~isdirs);
files = {files.name};
files = cellfun2(@(x)[dirpath '/' x],files);

if exist('extensions', 'var'),
  if ~iscell(extensions),
    extensions = {'.jpg', '.png', '.gif', '.ppm', '.tif', '.jpeg'};
  end
  ext = cellfun2(@(x)get_extension(x), files);
  ext = cellfun2(@(x)any(strcmp(x, extensions)), ext);
  ext = arrayfun(@(x)x{1}, ext);
  files = files(find(ext));
end

function extension = get_extension(file)
[~,~,extension] = fileparts(file);
