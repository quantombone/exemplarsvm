function make_lustre_mats
VOCinit;

myRandomize;
bg = get_pascal_bg('both');
rrr = randperm(length(bg));
bg = bg(rrr);
sbin = 8;
lpo = 10;
for i = 1:length(bg)
  [a,curid,tmp] = fileparts(bg{i});
  filer = sprintf('/lustre/tmalisie/VOC2007mats/%s.mat',curid);
  filerlock = [filer '.lock'];
  if fileexists(filer) || (mymkdir_dist(filerlock)==0)
    continue
  end
  I = convert_to_I(bg{i});
  [f,s] = featpyramid2(I,sbin,lpo);
  f2 = featpyramid2(flip_image(I),sbin,lpo)
  sizeI = size(I);
  save(filer,'f','f2','s','sizeI','sbin','lpo');
  rmdir(filerlock);
end