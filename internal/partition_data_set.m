function [positive_set, negative_set] = partition_data_set(data_set, ...
                                                  cls);
%split data into positive set and negative set
valids = cellfun(@(x)isfield(x,'objects'),data_set);
data_set = data_set(valids);
goods = cellfun(@(x)find(( ismember({x.objects.class},cls) & ...
                           ([x.objects.truncated]==0) & ...
                           ([x.objects.difficult]==0))),...
                data_set,'UniformOutput',false);

data_set = cellfun(@(x,y)setfield(x,'objects',x.objects(y)),data_set, ...
                  goods,'UniformOutput',false);

%prune empties
pos_length = cellfun(@(x)numel(x.objects)>0,data_set);
positive_set = data_set(find(pos_length));
negative_set = data_set(find(pos_length)==0);
