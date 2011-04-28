function [x, hg_size, coarse_boxes, allbads] = hog_from_bbox(I,bbox,params,mask)
%% Given an image and a bounding box in continuous coordinates,
%% we crop the image to the tighest fitting box
%% Tomasz Malisiewicz (tomasz@cmu.edu)

%try to extend box
H = bbox(4)-bbox(2)+1;
W = bbox(3)-bbox(1)+1;
bbox(1) = bbox(1) - .2*W;
bbox(3) = bbox(3) + .2*W;
bbox(2) = bbox(2) - .2*H;
bbox(4) = bbox(4) + .2*H;
bbox([1 3]) = cap_range(round(bbox([1 3])),1,size(I,2));
bbox([2 4]) = cap_range(round(bbox([2 4])),1,size(I,1));

I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
if exist('mask','var')
  mask = mask(bbox(2):bbox(4),bbox(1):bbox(3));
end

sbin = params.sbin;

%this controls how large the returned HOG templates are
MAX_CELL_DIM = params.MAX_CELL_DIM;
MIN_CELL_DIM = params.MIN_CELL_DIM;

[sss] = size(I);
sss = sss(1:2);

factors = [sbin*MAX_CELL_DIM/max(sss) ...
           sbin*MIN_CELL_DIM/min(sss)];

factor = max(factors);
news = round(size(I)*factor);

x = cell(0,1);

%% Here we define how large the Aspect-Ratio Wiggles should be
FRAC_CROP = .05;
NUM_CROPS_PER_DIM = 5;
cx = round(linspace(0,size(I,1)*FRAC_CROP,...
                    NUM_CROPS_PER_DIM));
cy = round(linspace(0,size(I,2)*FRAC_CROP,...
                    NUM_CROPS_PER_DIM));
cx = [0 0 0 0 0];
cy = [0 0 0 0 0];
III = cell(0,1);
allbads = [];
for a = 1:length(cx)
  for b = 1:length(cy)
    %crop boundaries resize to new size

    I2 = imresize(I(1+cx(a):end-cx(a),...
                    1+cy(b):end-cy(b),:),...
                  news(1:2));
    %I2 = imresize(I,news(1:2));
    %r = 1 - .6*rand;
    %I2 = imresize(I2,r);
    %I2 = imresize(I2,news(1:2));
    %fprintf(1,'.');
    I2 = max(0.0,min(1.0,I2));
    III{end+1} = I2;
    f = features(I2, sbin);
    
    if exist('mask','var')
      %mask2 = imresize(mask(1+cx(a):end-cx(a),...
      %                      1+cy(b):end-cy(b),:),...
      %                 news(1:2),'nearest');
      mask2 = imresize(mask,news(1:2),'nearest');
      mask2 = max(0.0,min(1.0,mask2));
      
      newbb = (size(f)+2)*sbin;
      newbb = [1 1 newbb(2) newbb(1)];
      %I3 = pad_image(I2,100);
      mask3 = pad_image(mask2,100);
      newbb = newbb+100;
      %I3 = I3(newbb(2):newbb(4),...
      %        newbb(1):newbb(3),:);
      
      mask3 = mask3(newbb(2):newbb(4),...
                    newbb(1):newbb(3),:);
      mask4 = imresize(mask3,1/sbin);
      mask4 = pad_image(mask4,-1);
      %bads = (mask4>.5);
      mask4 = max(0.0,mask4);
      allbads(:,:,end+1) = mask4;
      %allbads{end+1} = mask4;  
    end
    
    %zero out missing region
    %f(repmat(bads,[1 1 31]))=0;

    x{end+1} = f(:);
  end
end
allbads = allbads(:,:,2:end);

%enable to show mean image
if 0
  III = mean(cat(4,III{:}),4);
  imagesc(III)
  title('Mean Image');
  drawnow
  pause
end

x = cat(2,x{:});

% figure out the actual bb used (because it can spill outside/inside
% image boundary)
newbb = (size(f)+2)*sbin/factor;
newbb = [1 1 newbb(2) newbb(1)];

coarse_boxes = (repmat(newbb, length(cx)*length(cy), 1));

hg_size(1) = size(f,1);
hg_size(2) = size(f,2);

if min(hg_size) < 1
  error('error min hg size too small');
end

coarse_boxes(:,[1 3]) = coarse_boxes(:,[1 3]) + bbox(1)-1;
coarse_boxes(:,[2 4]) = coarse_boxes(:,[2 4]) + bbox(2)-1;
