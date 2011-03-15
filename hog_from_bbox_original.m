function [x, hg_size, coarse_box] = hog_from_bbox_original(I,bbox,params)
%% Given an image and a bounding box in continuous coordinates,
%% we crop the image to the tighest fitting box
%% Tomasz Malisiewicz (tomasz@cmu.edu)

Isize = size(I);
Isize = Isize(1:2);

% Get initial dimensions
w = bbox(3)-bbox(1)+1;
h = bbox(4)-bbox(2)+1;

if (w*h)*100 / prod(Isize) < 1
  %% try to increment by 10%
  bbox = bbox + round([-w -h w h]*.1);
  [x, hg_size, coarse_box] = hog_from_bbox_original(I,bbox,params);
  return;
end

%get center of GT box (we will align centers later)
wmid = round((bbox(3)+bbox(1)+1)/2);
hmid = round((bbox(4)+bbox(2)+1)/2);

w1 = w;
h1 = h;
aspect = h / w;
area = h * w;    

area = max(min(area, 5000), 3000);

% pick dimensions
w = sqrt(area/aspect);
h = w*aspect;

sbin = 8;
hg_size = [round(h/sbin) round(w/sbin)];

%make even sizes, makes for an easier job getting the lcm
hg_size = max(1,round(hg_size/2))*2;

%% find new bbox with the exact ratio and exact area
newaspect = hg_size(1) / hg_size(2);

%% needs to be a multiple of the least-common-multiple
multer = lcm(hg_size(2),sbin);

neww = multer:multer:3000;
newh = newaspect*neww;

goods = find(((round(newh) - newh)==0) & ((round(neww) - neww)==0));

neww = neww(goods);
newh = newh(goods);

olda = w1*h1;
newa = newh.*neww;

subs = find(neww >= w1 & newh >= h1);
neww = neww(subs);
newh = newh(subs);
newa = newa(subs);

%% We choose dimensions such that the GT box is enclosed WITHIN
%the coarse_box, but we find the tightest box
[aa,bb] = min(abs(newa - olda));
newh = newh(bb);
neww = neww(bb);

up = round([hmid-newh/2 wmid-neww/2]);
newb = [up(2) up(1) up(2)+neww-1 up(1)+newh-1];

os = getosmatrix_bb(newb,bbox);
fprintf(1,'OS of new region with GT region %.3f\n',os);

if 0
  figure(1)
  clf
  imagesc(I)
  axis image
  plot_bbox(bbox)
  hold on;
  plot_bbox(newb,'',[1 0 0],[1 0 0]);
  title(sprintf('%s.%d OS %.3f, cellsize = [%d x %d]\n',curid,objectid,os,hg_size(1),hg_size(2)));
end

extras = 500;
I2 = pad_image(I,extras);

%% Save the coarse box before we pad it
coarse_box = newb;
newb = newb+extras;


sbin2 = (newb(3)-newb(1)+1) / hg_size(2);

curx = (newb(2)-sbin2):(newb(4)+sbin2);
cury = (newb(1)-sbin2):(newb(3)+sbin2);
padI = zeros(length(curx),length(cury),3);
goodx = find(curx>=1 & curx<=size(I2,1));
goody = find(cury>=1 & cury<=size(I2,2));
padI(goodx,goody,:) = I2(curx(goodx),cury(goody),:);

%old one
%subI = I2(newb(2):newb(4),newb(1):newb(3),:);
%padI = (pad_image(subI,sbin2));

%clear the model
%clear m model

% here we save 25 little wiggles around the ground truth region
x = zeros(prod(hg_size)*31,0);
coarse_boxes = zeros(0,4);

for cx = -2:2
  for cy = -2:2
    curI = circshift(padI,[cx cy]);
    f = features(curI,sbin2);
    cb = coarse_box;
    cb([1 3]) = cb([1 3]) + cy;
    cb([2 4]) = cb([2 4]) + cx;
    coarse_boxes(end+1,:) = cb; 
    x(:,end+1) = f(:);
  end
end
