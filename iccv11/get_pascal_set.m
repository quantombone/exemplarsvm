function bg = get_pascal_set(VOCopts, target_directory, class)
% Get a subset of PASCAL dataset (as defined in VOCopts) from the
% 'train', 'trainval', or 'test' subsets ('both' refers to
% 'trainval'+'test')
% if class is "", choose all images in the subset
% if class is "+motorcycle" choose only images containing motorcycle
% if class is "-car" chose only images not containing cars
% returns:
%   [bg]: a cell array of images, such that I=convert_to_I(bg{i});

if (nargin == 2) && strcmp(target_directory,'both')
  bg = cat(1, ...
           get_pascal_set(VOCopts, 'trainval'),...
           get_pascal_set(VOCopts, 'test'));
  return;  
end

has_marker = (target_directory=='+') + ...
    (target_directory=='-');

has_marker = find(has_marker);
if length(has_marker) > 0
  
  if exist('class','var') && length(class)>0
    error('Cannot give directory with +/- sign AND class');
  end
  
  td = target_directory(1:has_marker(1)-1);
  cl = target_directory(has_marker(1):end);
  bg = get_pascal_set(VOCopts, td, cl);
  return;
end

if ~exist('target_directory','var');
  target_directory = 'train';
end

if ~exist('class','var') | (length(class)==0)
  [neg_set,gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),...
                          '%s %d');
  
elseif (class(1) == '-')
  class = class(2:end);
  [neg_set,gt] = textread(sprintf(VOCopts.clsimgsetpath,...
                                  class,target_directory),...
                          '%s %d');
  neg_set = neg_set(gt==-1);
  
elseif (class(1) == '+')
  class = class(2:end);
  [neg_set,gt] = textread(sprintf(VOCopts.clsimgsetpath,...
                                  class,target_directory),...
                          '%s %d');
  neg_set = neg_set(gt==1);
else
  error(sprintf(['Invalid class %s passed to get_pascal_set: must' ...
                 ' start with plus or minus sign'], class));
end

bg = cellfun2(@(x)sprintf(VOCopts.imgpath,x),neg_set);
