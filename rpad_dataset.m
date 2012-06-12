function data_set = rpad_dataset(data_set)
frac = .2;

for i = 1:length(data_set)
  data_set{i}.I = @()rotate_pad(toI(data_set{i}.I),frac);
  obj = data_set{i}.objects;
  isize = data_set{i}.imgsize;
  for k = 1:length(obj)
    bb = data_set{i}.objects(k).bbox;
    bb([1 3]) = bb([1 3]) + isize(1)*frac;
    bb([2 4]) = bb([2 4]) + isize(2)*frac;

    data_set{i}.objects(k).bbox = round(bb);
    
    
  end
end
