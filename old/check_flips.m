function check_flips


sizeI = [1000 1000 3];

for q = 1:1000
  b = 200+600*rand([1 4]);
  W = b(4)-b(2)+1;
  H = b(3)-b(2)+1;
  
  if W<=0 || H<=0
    continue    
  end
  
  b2 = flip_box(b,sizeI);
  b3 = flip_box(b2,sizeI);
  
  norm(b - b3)

end