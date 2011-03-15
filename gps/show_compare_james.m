function show_compare_james
%Here we show both results to compare (before blending)
%will it blend?

%VOCinit;

BASEDIR = ['/nfs/baikal/jhhays/scene_completion/' ...
           'old_inpainting_matches/'];

IMGDIR = '/nfs/baikal/jhhays/scene_completion/old_inpainting_imgs/';
files = dir([BASEDIR '/*mat']);

load /nfs/baikal/tmalisie/will-it-blend/james_bbs.mat 

TOPK = 3;

for i = 1:length(files)
  res = load([BASEDIR files(i).name]);
  ending = strfind(files(i).name,'_query_info.mat');
  filepart = files(i).name(1:ending-1);
  imfile = dir([IMGDIR filepart '.*']);
  imfile = [IMGDIR imfile(1).name];
  
  maskfile = dir([IMGDIR filepart '_mask.*']);
  maskfile = [IMGDIR maskfile(1).name];

  I = im2double(imread(imfile));
  mask = imread(maskfile);
  mask = (mask(:,:,1)~=0);
  I(find(repmat(mask,[1 1 3])))=0;
  
  figure(1)
  clf
  imagesc(I);
  axis image
  axis off
  title(sprintf('Query image %d',i))
  
  figure(2)
  clf
  for j = 1:TOPK
    subplot(3,TOPK,j)
    I = load_james_image(res.best_indices(j));
    imagesc(I)
    title(sprintf('James Match %d',j));
    axis image
    axis off
  end
  [aa,bb] = sort(james_bbs{i}(:,end),'descend');
  ims = james_bbs{i}(bb,5);
  
  for j = 1:min(TOPK,length(bb))
    I = load_james_image(res.best_indices(ims(j)));
    I = im2double(I);
    I = capsize(I);

    subplot(3,TOPK,j+TOPK)
    imagesc(I)
    title(sprintf('TJM Match %d',j));
    plot_bbox(james_bbs{i}(bb(j),1:4))
    axis image
    axis off
  end
  
  for j = 1:min(TOPK,length(bb))
    I = load_james_image(res.best_indices(ims(j)));
    I = im2double(I);
    I = capsize(I);
    I = pad_image(I,300);
    curbb = james_bbs{i}(bb(j),1:4);
    curbb = round(curbb+300);
    I = I(curbb(2):curbb(4),curbb(1):curbb(3),:);
    subplot(3,TOPK,j+2*TOPK)
    imagesc(I)
    title(sprintf('TJM subimage Match %d',j));
    axis image
    axis off
  end
  
  %keyboard
  %drawnow
  %set(gcf,'PaperPosition',[0 0 10 10])
  %print(gcf,'-dpng',[VOCopts.dumpdir '/james' num2str(i) '.png']);
  pause
end

function I = capsize(I)
MAXSIZE = 300;
if max(size(I))>MAXSIZE
  factor = MAXSIZE/max(size(I));
  I = max(0.0,min(1.0,imresize(I,factor)));
end
