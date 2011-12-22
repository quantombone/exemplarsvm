function show_james;

%VOCinit;

BASEDIR = ['/nfs/baikal/jhhays/scene_completion/' ...
           'old_inpainting_matches/'];

IMGDIR = '/nfs/baikal/jhhays/scene_completion/old_inpainting_imgs/';
files = dir([BASEDIR '/*mat']);
for i = 1:length(files)
  res = load([BASEDIR files(i).name]);
  ending = strfind(files(i).name,'_query_info.mat');
  filepart = files(i).name(1:ending-1);
  imfile = dir([IMGDIR filepart '.*']);
  imfile = [IMGDIR imfile(1).name];
  
  maskfile = dir([IMGDIR filepart '_mask.*']);
  maskfile = [IMGDIR maskfile(1).name];
  subplot(5,5,1)
  I = im2double(imread(imfile));
  mask = imread(maskfile);
  mask = (mask(:,:,1)~=0);
  I(find(repmat(mask,[1 1 3])))=0;
  imagesc(I);
  axis image
  axis off
  title(sprintf('Query image %d',i))
  for j = 1:24
    subplot(5,5,j+1)
    I = load_james_image(res.best_indices(j));
    imagesc(I)
    axis image
    axis off
  end
  %keyboard
  %drawnow
  %set(gcf,'PaperPosition',[0 0 10 10])
  %print(gcf,'-dpng',[VOCopts.dumpdir '/james' num2str(i) '.png']);
end
