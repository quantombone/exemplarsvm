function initialize_james_exemplars;
% Initialize exemplars from James Hays' SIGGRAPH scene completion paper

% Tomasz Malisiewicz (tomasz@cmu.edu)

landmark_name = 'james';

%where james stored his mat files
BASEDIR = ['/nfs/baikal/jhhays/scene_completion/' ...
           'old_inpainting_matches/'];

%where james stored his query images
IMGDIR = '/nfs/baikal/jhhays/scene_completion/old_inpainting_imgs/';
files = dir([BASEDIR '/*mat']);

image_names = cell(length(files),1);
mask_names = cell(length(files),1);

for i = 1:length(files)
  ending = strfind(files(i).name,'_query_info.mat');
  filepart = files(i).name(1:ending-1);
  
  %we don't know if the image is a png,bmp,gif, or whatnot, so use dir
  %command to list files which match suffix
  
  imfile = dir([IMGDIR filepart '.*']);
  imfile = [IMGDIR imfile(1).name];
  
  maskfile = dir([IMGDIR filepart '_mask.*']);
  maskfile = [IMGDIR maskfile(1).name];
  

  image_names{i} = imfile;
  mask_names{i} = maskfile;
end


ALLOW_SELECTION = 0;

VOCinit;

if ~exist('N_PER_FRAME','var')
  N_PER_FRAME = 1;
  fprintf(1,'Defaulting to %d regions per image\n', N_PER_FRAME);
end

results_directory = ...
    sprintf('%s/exemplars/',VOCopts.localdir);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

BYPASS_SELECTION = 1;
N_PER_FRAME = 1;

for i = 1:length(image_names)  
  
  filer = sprintf('%s/%s.%05d-%d.mat',results_directory,...
                  landmark_name, i, N_PER_FRAME);
  if fileexists(filer)
    continue
  end
  fprintf(1,'Processing %d/%d\n',i,length(image_names));
  
  I = convert_to_I(image_names{i});
  mask = imread(mask_names{i});
  mask = double(mask(:,:,1)~=0);

  % %% Make sure the maximum image dimensions is 600 
  % ms = max(size(I));
  % if ms > 600
  %   I = min(1.0,max(0.0,imresize(I,600/ms)));
  % end
  
  mask = imresize(mask,[size(I,1) size(I,2)],'nearest');


  for objectid = 1:N_PER_FRAME
    
    filer = sprintf('%s/%s.%05d-%d.mat',results_directory,...
                    landmark_name, i, objectid);
    if fileexists(filer)
      continue
    end

    if BYPASS_SELECTION == 0
    
    %% Get selection regions
    while 1
      figure(1)
      clf
      imagesc(I)
      axis image
      axis off
      title(sprintf('IM %d/%d, Select Rectangular Region %d/%d:', i,...
                    length(image_names), objectid, N_PER_FRAME));
      fprintf(1,['Click a corner, hold until diagonally opposite corner,' ...
                 ' and release\n']);
      h = imrect;
      bbox = getPosition(h);
      
      if (bbox(3)*bbox(4) < 50)
        fprintf(1,'Region too small, try again\n');
      else
        break;
      end
    end
      
    bbox(3) = bbox(3) + bbox(1);
    bbox(4) = bbox(4) + bbox(2);
    bbox = round(bbox);
    
    plot_bbox(bbox)
    pause(.1)
    
    else
      %BYPASSING SELECTIONS
      bbox = [1 1 size(I,2) size(I,1)];
    end
    
    %Get the hog features (+wiggles) from the specified bounding box
    clear model;
    
    model.params.sbin = 10;
    model.params.MAX_CELL_DIM = 12;
    model.params.SVMC = .0001;
    
    OUTSIDE_BOX = 0;

    [model.x, model.hg_size, model.coarse_box, bad_zones] = ...
        hog_from_bbox(I, bbox, model.params, OUTSIDE_BOX, mask);
    
    mean_bad_zone = mean(bad_zones,3)>.5;
    
    %% the good mask tells us the active "legitimate region"
    model.mask = 1-mean_bad_zone;
    
    model.x(find(repmat(mean_bad_zone,[1 1 31])),:) = 0;
        
    model.hg_size = [model.hg_size 31];
    model.w = reshape(mean(model.x,2),model.hg_size);
    %make mean 0
    model.w = model.w - mean(model.w(:));
    model.b = 0;
    
    if 0
    figure(1)
    subplot(1,2,1)
    imagesc(I.*repmat(1-mask,[1 1 3]))
    subplot(1,2,2)
    imagesc(HOGpicture(pad_image(model.w,1)))
    pause
    end
    
    model.nsv = zeros(prod(model.hg_size),0);
    model.svids = [];
    model.mine_ids = [];
    
    m.models_name = 'james_models';
    m.gt_box = bbox;
    m.model = model;
    m.bg = sprintf('get_arg_2(@()get_james_nearest(1000,''%s''))',...
                   landmark_name);

    m.fg = sprintf('get_james_160matches(%d)',i);
        
    m.index = i;
    m.I = I;
    m.cls = landmark_name;

    save(filer,'m');
  end  
end
