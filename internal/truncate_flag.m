function d2 = truncate_flag(d2)
c = 0;
for i = 1:length(d2)
  for j = 1:length(d2{i}.objects)

    os = getosmatrix_bb(d2{i}.objects(j).bbox, ...
                        d2{i}.objects(j).gt_box);


    if (length(os) == 0) || os < .5
      d2{i}.objects(j).difficult = 1;
      c = c + 1;
    end

  end
end

fprintf(1,'made %d objects difficult\n',c);

