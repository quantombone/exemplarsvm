function initialize_exemplars_dir(sketchdir, N_PER_FRAME)
% Initialize script which given a directory name, loads all images
% from directory, and lets the user select N rectangular "exemplar"
% regions in each image
% sketchdir:      The directory containing some images files
% [N_PER_FRAME]:  The number of regions to select per image (Defaults to 1)

% Tomasz Malisiewicz (tomasz@cmu.edu)

ALLOW_SELECTION = 0;
f = dir(sketchdir);

for i = 3:length(f)
  initialize_exemplars_dir_driver([sketchdir '/' f(i).name], N_PER_FRAME, f(i).name);
end

function initialize_exemplars_dir_driver(sketchdir, N_PER_FRAME, landmark_name)

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

% results_directory = sprintf('%s/%s', results_directory, landmark_name);
% if ~exist(results_directory,'dir')
%   fprintf(1,'Making directory %s\n',results_directory);
%   mkdir(results_directory);
% end

%% only consider .jpg or .png or .gif files inside dir
endings = {'jpg','png','gif','JPEG','JPG'};
files = cellfun2(@(x)dir([sketchdir '/*.' x]), endings);
files = cat(1,files{:});
image_names = cellfun2(@(x)[sketchdir '/' x],{files.name});

BYPASS_SELECTION = 0;
if N_PER_FRAME == 0
  BYPASS_SELECTION = 1;
  N_PER_FRAME = 1;
end

for i = 1:length(image_names)  
  Ibase = convert_to_I(image_names{i});
  padder = round(mean([size(Ibase,1) size(Ibase,2)])*.1);
  Ibase = pad_image(Ibase,padder,1);
  %imagesc(Ibase)
  %drawnow


  %keyboard
  %% Make sure the maximum image dimensions is 600 
  ms = max(size(Ibase));
  if ms > 600
    Ibase = min(1.0,max(0.0,imresize(Ibase,600/ms)));
  end

  for objectid = 1:N_PER_FRAME
    
    filer = sprintf('%s/%s.%d-%d.mat',results_directory,landmark_name,i,objectid);
    if fileexists(filer)
      continue
    end

    if BYPASS_SELECTION == 0
    
    %% Get selection regions
    while 1
      figure(1)
      clf
      imagesc(Ibase)
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
      bbox = [1 1 size(Ibase,2) size(Ibase,1)];
    end
    
    I = Ibase;

    %Get the hog features (+wiggles) from the specified bounding box
    clear model;
    
    model.params.sbin = 10;
    model.params.MAX_CELL_DIM = 12;
    model.params.SVMC = .0001;
    
    %SVMC from CVPR11
    %model.params.SVMC = .01;    
    
    [model.x, model.hg_size, model.coarse_box] = ...
        hog_from_bbox(I, bbox, model.params); 
    model.hg_size = [model.hg_size 31];
    
    model.w = reshape(mean(model.x,2), model.hg_size);
    
    %make mean 0
    model.w = model.w - mean(model.w(:));
    
    model.b = 0;
    model.nsv = zeros(prod(model.hg_size),0);
    model.svids = [];
    model.mine_ids = [];
    
    m.models_name = 'sketch_models';
    m.gt_box = bbox;
    m.model = model;
    %m.bg = sprintf('get_arg_2(@()get_james_nearest(1000,''%s''))',...
    %               landmark_name);

    %m.fg = sprintf('get_arg_1(@()get_james_nearest(1000,''%s''))',...
    %               landmark_name);
    
    m.bg = sprintf(sprintf(['get_pascal_bg(''train'',''-' ...
                    '%s'')'],landmark_name));
    m.fg = sprintf(sprintf(['get_pascal_bg(''test'',''' ...
                    '%s'')'],landmark_name));
    
    
    m.index = i;
    m.I = I;
    m.cls = landmark_name;
    
    save(filer,'m');
  end  
end
