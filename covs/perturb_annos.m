function data_set = perturb_annos(data_set)
for i = 1:length(data_set)
  % Isize = [eval(data_set{i}.imagesize.nrows) ...
  %          eval(data_set{i}.imagesize.ncols)]
  Isize = size(toI(data_set{i}));
  for j = 1:length(data_set{i}.objects)
    
    bb2 = data_set{i}.objects(j).bbox(1:4);
    F = 10;
    while 1
      diffs = F*[-.5+rand(1,4)];
      diffs = F*[-1 -1 1 1].*rand(1,4);
      bb2 = data_set{i}.objects(j).bbox(1:4)+diffs;
      if bb2(3)>bb2(1)+10 && bb2(4)>bb2(2)+10
        break;
      end
    end
    
    %bb2(4) = bb2(4)+50;
    data_set{i}.objects(j).bbox(1:4) = bb2;
      
    data_set{i}.objects(j).bbox = ...
        clip_to_image(data_set{i}.objects(j).bbox,[1 1 Isize(2) Isize(1)]);
  end
end