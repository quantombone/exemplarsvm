function [positive_set,negative_set,stripped_set] = get_objects_set(data_set, cls)
%Extract positives from the data set

%Skip truncated and difficult ones

has_truncated = cellfun(@(x)isfield(x.objects,'truncated'),data_set);
has_difficult = cellfun(@(x)isfield(x.objects,'difficult'),data_set);

if any(has_truncated)  || any(has_difficult)
  good_objects = cellfun(@(x)find(( ismember({x.objects.class},cls) & ...
                                    ([x.objects.truncated]==0) & ...
                                    ([x.objects.difficult]==0))),...
                         data_set,'UniformOutput',false);
else
  good_objects = cellfun(@(x)find(ismember({x.objects.class},cls)),...
                                  data_set,'UniformOutput',false);
end

bads = find(cellfun(@(x)length(x)==0,good_objects));

for j = 1:length(bads)
  good_objects{bads(j)} = [];
end

good_images = cellfun(@(x)numel(x)>0, good_objects);
stripped_set = cellfun(@(x,y)setfield(x,'objects',x.objects(y)),...
                       data_set, ...
                       good_objects,'UniformOutput', ...
                       false);
positive_set = stripped_set(good_images);

bad_images = cellfun(@(x)length(find(( ismember({x.objects.class},cls)))==0)>0,data_set);

negative_set = data_set(bad_images);
