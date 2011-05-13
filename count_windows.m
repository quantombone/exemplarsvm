function count_windows
%%Here we count how many windows there are in a slide!!!

bg = get_pascal_bg('trainval');
I = convert_to_I(bg{1});
sbin = 8;
lpo = 10;
padder = 10;

%hg_size = [20 20];
p = featpyramid2(I,sbin,lpo);
p = cellfun2(@(x)padarray(x,[2 2 0]),p);
p = cellfun2(@(x)[size(x,1) size(x,2)],p);

for r = 1:20
  hg_size = [r r];
  p2 = cellfun2(@(x)max(0.0,x-hg_size),p);
  p2 = cellfun(@(x)prod(x),p2);
  nwin(r) = sum(p2);
end


keyboard
