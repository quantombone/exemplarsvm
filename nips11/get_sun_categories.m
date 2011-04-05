function [y,catnames] = get_sun_categories(ids)
%compute classes from filenames for SUN397 dataset

ids2 = cellfun2(@(x)x((strfind(x,'SUN397')+6):end),ids);
slashes = cellfun2(@(x)strfind(x,'/'),ids2);
slashes = cellfun(@(x)x(end),slashes);
cats = ids2;
for i = 1:length(ids2)
  cats{i} = ids2{i}(1:slashes(i));
end
catnames = unique(cats);
[aa,y] = ismember(cats,catnames);

