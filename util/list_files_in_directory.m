function files = list_files_in_directory(dirpath, extensions)
% Given a directory, list all files inside the directory and create a cell
% array of strings, where each string is the file location
%
% extensions is optional and can take a variety of types:
% - if it is a cell array, only return files with extensions in this array
%     >> files = list_files_in_directory('~/myimages/',{'.jpg','.png'});
% - if it is the string 'images', only return images
%     >> files = list_files_in_directory('~/myimages/','images')
% - if it is a string, only return files with this extension
%     >> files = list_files_in_directory('~/myimages/','.jpg')
% - if it is not given, return all files
%     >> files = list_files_in_directory('~/myimages/','.jpg')
%
% NOTE: this can be used with convert_to_I, as follows:
%     >> files = list_files_in_directory('~/myimages/');
%     >> imagesc(convert_to_I(files{10}))

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
  if isstr(extensions),
    if strcmpi(extensions, 'images') || strcmpi(extensions, 'image'),
      extensions = {'.jpg', '.png', '.gif', '.ppm', '.tif', '.jpeg'};
    elseif strcmpi(extensions, 'videos') || strcmpi(extensions, 'video'),
      extensions = {'.mp4', '.mov', '.avi', '.ogv', '.wmv'};
    else,
      extensions = {extensions};
    end
  end
  ext = cellfun2(@(x)get_extension(x), files);
  ext = cellfun2(@(x)any(strcmpi(x, extensions)), ext);
  ext = arrayfun(@(x)x{1}, ext);
  files = files(find(ext));
end

function extension = get_extension(file)
[~,~,extension] = fileparts(file);
