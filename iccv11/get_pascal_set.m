function bg = get_pascal_set(VOCopts, target_directory, class)
% Get a subset of PASCAL dataset (as defined in VOCopts) from the
% 'train', 'trainval', or 'test' subsets ('both' refers to
% 'trainval'+'test')
% if class is "", choose all images in the subset
% if class is "motorcycle" choose only images containing motorcycle
% if class is "-car" chose only images not containing cars
% returns:
%   [bg]: a cell array of images, such that I=convert_to_I(bg{i});

if (nargin == 2) && strcmp(target_directory,'both')
  bg = cat(1, ...
           get_pascal_bg(VOCopts, 'trainval'),...
           get_pascal_bg(VOCopts, 'test'));
  return;  
end

if ~exist('target_directory','var');
  target_directory = 'train';
end

%% eliminating images with id kill_image_id (if present)
%if ~exist('kill_image_id','var')
%  kill_image_id = '';
%end

if ~exist('class','var') | (length(class)==0)
  [neg_set,gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),...
                          '%s %d');
elseif (class(1) == '-')
  minus = 1;
  class = class(2:end);
  [neg_set,gt] = textread(sprintf(VOCopts.clsimgsetpath,...
                                  class,target_directory),...
                          '%s %d');
  neg_set = neg_set(gt==-1);
else
  [neg_set,gt] = textread(sprintf(VOCopts.clsimgsetpath,...
                                  class,target_directory),...
                          '%s %d');
  neg_set = neg_set(gt==1);
end

%if exist('kill_image_id','var')
%  neg_set = setdiff(neg_set,kill_image_id);
%end

bg = cellfun2(@(x)sprintf(VOCopts.imgpath,x),neg_set);
