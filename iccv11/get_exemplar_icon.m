function [Iex,alphamask,Icb] = get_exemplar_icon(models,index,flip,subind)
%Extract an exemplar visualization image (one from gt box, one from
%cb box) [and does flip if specified]
%Tomasz Malisiewicz (tomasz@cmu.edu)

%Subind indicates which window to show (defaults to the base)
if ~exist('subind','var')
  subind = 1;
end

if ~exist('flip','var')
  flip = 0;
  if isfield(models{index},'FLIP_LR') && models{index}.FLIP_LR==1
    flip = 1;
  end
end

cb = models{index}.model.target_id{subind}.bb;
d1 = max(0,1 - cb(1));
d2 = max(0,1 - cb(2));
d3 = max(0,cb(3) - models{index}.sizeI(2));
d4 = max(0,cb(4) - models{index}.sizeI(1));
mypad = max([d1,d2,d3,d4]);
PADDER = round(mypad)+2;


%PADDER = 100;

if isfield(models{index},'I')
  I = models{index}.I;
  
  cb = models{index}.gt_box;    
  Iex = pad_image(I, PADDER);
  cb = round(cb + PADDER);
  Iex = Iex(cb(2):cb(4),cb(1):cb(3),:);
  
  %cb = models{index}.model.coarse_box;
  cb = models{index}.model.target_id{subind}.bb;
  Icb = pad_image(I, PADDER);
  cb = round(cb + PADDER);
  Icb = Icb(cb(2):cb(4),cb(1):cb(3),:);
  
else

  try
    VOCinit;
    I = im2double(imread(sprintf(VOCopts.imgpath, ...
                                   models{index}.curid)));
    %fprintf(1,'warning using coarse box\n');
    cb = models{index}.gt_box;    
    Iex = pad_image(I, PADDER);
    cb = round(cb + PADDER);
    Iex = Iex(cb(2):cb(4),cb(1):cb(3),:);
    
    %cb = models{index}.model.coarse_box;
    cb = models{index}.model.target_id{subind}.bb;
    Icb = pad_image(I, PADDER);
    cb = round(cb + PADDER);
    Icb = Icb(cb(2):cb(4),cb(1):cb(3),:);
  catch
    keyboard
    %if cannot do the step, then create a hog picture visualization
    %which can also be displayed with imagesc
    Iex = HOGpicture(models{index}.model.w);
    Iex = jettify(Iex);
    Icb = Iex;
  end
end

if flip == 1
  fprintf(1,'flipping\n');
  Iex = flip_image(Iex);
  Icb = flip_image(Icb);
end

alphamask = zeros(size(Iex,1), size(Iex,2));
