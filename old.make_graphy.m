function G = make_graphy(models)

names = cellfun2(@(x)sprintf('%s.%d',x.curid,x.objectid),models);

G = zeros(length(names),length(names));

for i = 1:length(models)
  a = i;
  for j = 1:length(models{i}.friend_info)
    ud = unique(models{i}.friend_info{j}.os_id);
    for k = 1:length(ud)
      curname = sprintf('%s.%d',models{i}.friend_info{j}.curid,...
                        ud(k));
      b = find(ismember(names,curname));
      G(a,b) = 1;
    end
  end
end
 