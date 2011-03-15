function [Iex,alphamask] = get_exemplar_icon(models,index)
%given a cell array of models, show model at index

flip = 0;
if isfield(models{index},'FLIP_LR') && models{index}.FLIP_LR==1
  flip = 1;
end

if isfield(models{index},'I')
  Iex = models{index}.I;
else


try
  VOCinit;
  Iex = im2double(imread(sprintf(VOCopts.imgpath, ...
                                 models{index}.curid)));
  fprintf(1,'warning using coarse box\n');
  cb = models{index}.gt_box;

  cb = models{index}.model.coarse_box;

  PADDER = 100;

  Iex = pad_image(Iex, PADDER);
  cb = round(cb + PADDER);
  Iex = Iex(cb(2):cb(4),cb(1):cb(3),:);
  
catch
  %Iex = zeros(100,100,3);
  Iex = HOGpicture(models{index}.model.w);
  Iex = Iex - min(Iex(:));
  Iex = Iex / max(Iex(:));
  
  colors = jet(100);
  c = colors(round(Iex(:)*99+1),:);
  c = reshape(c,[size(Iex,1) ...
                    size(Iex,2) 3]);
  Iex = c;
end
end


if flip == 1
  fprintf(1,'flipping\n');
  Iex = flip_image(Iex);
end

alphamask = zeros(size(Iex,1), size(Iex,2));