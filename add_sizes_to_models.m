function models = add_sizes_to_models(models)
%Load up the image for each model and load it to get dimensions
VOCinit;

for exid = 1:length(models)
  if ~isfield(models{exid},'sizeI');
    fprintf(1,'!');
    
    Iex = im2double(imread(sprintf(VOCopts.imgpath, ...
                                   models{exid}.curid)));
    g = models{exid}.gt_box;
    Igt = Iex(g(2):g(4),g(1):g(3),:);
    
    newsize = models{exid}.model.hg_size(1:2)*8;
    Igt = imresize(Igt,newsize);
    Igt = max(0.0,min(1.0,Igt));
    
    models{exid}.sizeI = size(Iex);
    %models{exid}.I = Igt;
    %if isfield(models{exid},'FLIP_LR') && models{exid}.FLIP_LR==1
    %  models{exid}.I = flip_image(models{exid}.I);
    %end
  end
end


