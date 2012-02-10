function [positive_set, negative_set] = ...
    get_positive_negative_sets(data_set, cls)
% Extract positive and negative datasets from the data_set variable
% according to the class specified in cls.
% Inputs: 
%   data_set: a cell array of image objects
%   cls: a single class to define the positive images
% Outputs:
%   positive_set: a set of images which contains only objects of
%      class cls, and each image contains at least one such object
%   negative_set: a set of negative images which do not contain any
%      instances of class cls

nonempties = cellfun(@(x)isfield(x,'objects'),data_set);
negative_set = data_set(~logical(nonempties));
data_set = data_set(logical(nonempties));

%Skip truncated and difficult objects
%has_truncated = cellfun(@(x)isfield(x.objects,'truncated'),data_set);
has_difficult = cellfun(@(x)isfield(x.objects,'difficult'),data_set);

if  any(has_difficult)
  %do not use truncated and difficult objects
  good_objects = cellfun(@(x)find(( ismember({x.objects.class},cls) & ...
                                    ([x.objects.difficult]==0))),...
                         data_set,'UniformOutput',false);
  
  %                                    ([x.objects.truncated]==0) & ...
else
  good_objects = cellfun(@(x)find(ismember({x.objects.class},cls)),...
                                  data_set,'UniformOutput',false);
end

bads = find(cellfun(@(x)length(x)==0,good_objects));

for j = 1:length(bads)
  good_objects{bads(j)} = [];
end

good_images = cellfun(@(x)numel(x)>0, good_objects);
stripped_set = [cellfun(@(x,y)setfield(x,'objects',x.objects(y)),...
                       data_set, ...
                       good_objects,'UniformOutput', ...
                       false) negative_set];

for k = 1:length(stripped_set)
  if ~isstruct(stripped_set{k})
    clear s
    s.I = stripped_set{k};
    s.objects = [];
    stripped_set{k} = s;
  elseif ~isfield(stripped_set{k},'objects')
    stripped_set{k}.objects = [];
  end
end


good_images = cellfun(@(x)numel(x.objects)>0, stripped_set);

positive_set = stripped_set(good_images);
negative_set = stripped_set(~good_images);
