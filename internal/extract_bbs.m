function bbs = extract_bbs(data_set)

bbs = cell(0,1);
for i = 1:length(data_set)
  if isfield(data_set{i},'objects')
    for j = 1:length(data_set{i}.objects)
      bb = data_set{i}.objects(j).bbox;
      bb(12) = 0;
      bb(11) = i;
      bbs{end+1} = bb;

    end
  end  
end

bbs = cat(1,bbs{:});