function dump_icons(g)

for i = 1:length(g)
  I = show_g(g,i,1);
  maxdim = max(size(I));
  I = imresize(I,200/maxdim);
  imwrite(I,sprintf('/nfs/baikal/tmalisie/cowicons/%05d.jpg',i));
  fprintf(1,'.');
  
end