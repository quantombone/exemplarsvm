function show_svs
VOCinit

BASER = 'normal.mined';
%BASER = 'mined';

BASEDIR = sprintf('/nfs/baikal/tmalisie/local/VOC2007/exemplars/%s/',BASER);
files = dir([BASEDIR '10.*.cow.mat']);

RESDIR = '/nfs/baikal/tmalisie/labelme400/www/siggraph/cowsvs/';
             
if ~exist(RESDIR,'dir')
  mkdir(RESDIR);
end
for i = 1:length(files)

  r = load([BASEDIR files(i).name]);

  

  I = imread(sprintf(VOCopts.imgpath,r.m.curid));
  figure(1)
  clf
  imagesc(I)
  axis image
  axis off
  plot_bbox(r.m.model.coarse_box)
  
  print(gcf,'-djpeg',sprintf('%s/%05d.jpeg',RESDIR,i));
  bg = eval(r.m.bg);

  figure(2)
  clf
  Isv = get_sv_stack(r.m.model.svids,bg);
  imagesc(Isv)
  imwrite(Isv,sprintf('%s/%05d_svs.%s.jpeg',RESDIR,i,BASER));

end