function I = show_g(g,index,lvl,offset,wsize)
VOCinit;

I = imread(sprintf(VOCopts.imgpath,g{index}.curid));
I = im2double(I);

box = round(g{index}.curb{lvl});

if ~exist('offset','var')
  I = I(box(2):box(4),box(1):box(3),:);
else
  
  H = (box(4) - box(2));
  W = (box(3) - box(1));
  
  
  miniW = W / size(g{index}.curw{lvl},2);
  miniH = H / size(g{index}.curw{lvl},1);

  box2([1 2]) = box([1 2]);
  box2([3 4]) = box2([1 2]) + [miniW*wsize(2) miniH*wsize(1)];
  box2([1 3]) = box2([1 3]) + miniW*offset(2);
  box2([2 4]) = box2([2 4]) + miniH*offset(1);
  
  I = pad_image(I,400);
  box2 = round(box2+400);
  I = I(box2(2):box2(4),box2(1):box2(3),:);
end

 