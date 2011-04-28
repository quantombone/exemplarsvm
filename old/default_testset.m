function [bg,setname] = default_testset;
%Get the default test set and its name

if 0
  %Get 10000 random images from the gps dataset
  bg = get_james_bg(10000,10000+(1:10000));
  setname = 'randjames';
  return;
end

%Load the entire PASCAL VOC 2007 trainval+test set
target_directory{1} = 'trainval';
target_directory{2} = 'test';
class = '';
curid = '';
for i = 1:length(target_directory)
  bgs{i} = get_pascal_bg(target_directory{i});
end
bg = cat(1,bgs{:});
setname = 'voc';
