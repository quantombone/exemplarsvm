function data_set = replace_class(data_set,cls)
strings = textscan(strrep(cls,'+',' '),'%s');
strings = strings{1};
if length(strings) == 1
  return;
end
  
for i = 1:length(data_set)
  if isfield(data_set{i},'objects')
    for j = 1:length(data_set{i}.objects)
      if ismember(data_set{i}.objects(j).class,strings)
        data_set{i}.objects(j).class = cls;
      end
    end
  end
end
