function grid = esvm_detect_imageset(imageset, models, ...
                                     params, setname)
% Apply a set of models (raw exemplars, trained exemplars, dalals,
% poselets, components, etc) to a set of images.  
%
% imageset: a (virtual) set of images, such that
%   convert_to_I(imageset{i}) returns an image
% models: Input cell array of models
% dataset_params(optional): detection parameters
% setname(optional): a name of the set, which lets us cache results

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).

  
if ~exist('params','var')
  params = esvm_get_default_mining_params;
end

%Only allow display to be enabled on a machine with X
display = 0;

save_files = 1;
if ~exist('setname','var')
  save_files = 0;
  params.detect_images_per_chunk = 1;
  setname = '';
end

if isempty(imageset)
  grid = {};
  return;
end

if save_files == 1
  
  final_file = sprintf('%s/applied/%s-%s.mat',...
                       params.dataset_params.localdir,setname, ...
                       models{1}.models_name);

  if fileexists(final_file)
    res = load(final_file);
    grid = res.grid;
    return;
  end
end

if display == 1
  fprintf(1,'DISPLAY ENABLED, NOT SAVING RESULTS!\n');
  params.detect_images_per_chunk = 1;
end

% if ~isfield(params.dataset_params,'params')
%   params = get_default_mining_params;
% else
%   params = params.dataset_params.params;
% end

if save_files == 1
  baser = sprintf('%s/applied/%s-%s/',params.dataset_params.localdir,setname, ...
                  models{1}.models_name);
else
  baser = '';
end

if (save_files==1) && (display == 0) && (~exist(baser,'dir'))
  fprintf(1,'Making directory %s\n',baser);
  mkdir(baser);
end

%% Chunk the data into detect_images_per_chunk images per chunk so that we
%process several images, then write results for entire chunk

inds = do_partition(1:length(imageset),params.detect_images_per_chunk);

% randomize chunk orderings
myRandomize;
ordering = randperm(length(inds));
if (display) == 1 || (save_files == 0)
  ordering = 1:length(ordering);
end

%[v,host_string]=unix('hostname');

allfiles = cell(length(ordering), 1);
counter = 0;
for i = 1:length(ordering)
  ind1 = inds{ordering(i)}(1);
  ind2 = inds{ordering(i)}(end);
  filer = sprintf('%s/result_%05d-%05d.mat',baser,ind1,ind2);
  allfiles{i} = filer;
  filerlock = [filer '.lock'];

  if save_files == 1
    if ~display && (fileexists(filer) || (mymkdir_dist(filerlock) == 0))
      continue
    end
  end
  res = cell(0,1);

  %% pre-load all images in a chunk
  %fprintf(1,'Preloading %d images\n',length(inds{ordering(i)}));
  clear Is;
  Is = cellfun2(@(x)convert_to_I(x),imageset(inds{ordering(i)}));

  %for j = 1:length(inds{ordering(i)})
  %  Is{j} = convert_to_I(imageset{inds{ordering(i)}(j)});
  %end
  L = length(inds{ordering(i)});

  for j = 1:L

    index = inds{ordering(i)}(j);
    fprintf(1,' --image %05d/%05d:',counter+j,length(imageset));
    Iname = imageset{index};
    %curid = -1;
    [tmp,curid,tmp] = fileparts(Iname);
    
    I = Is{j};
       
    starter = tic;
    rs = esvm_detect(I, models, params);

    
    % for q = 1:length(rs.bbs)
    %   if ~isempty(rs.bbs{q})
    %     rs.bbs{q}(:,11) = index;
    %     if length(rs.bbs{q}(1,:))~=12
    %       error('BUG: Invalid length bb');
    %     end
    %   end
    % end

    coarse_boxes = cat(1,rs.bbs{:});
    if ~isempty(coarse_boxes)
      coarse_boxes(:,11) = index;
      scores = coarse_boxes(:,end);
    else
      scores = [];
    end
    [aa,bb] = max(scores);
    fprintf(1,' %d exemplars took %.3fsec, #windows=%05d, max=%.3f \n',...
            length(models),toc(starter),length(scores),aa);
    
    % Transfer GT boxes from models onto the detection windows
    boxes = esvm_adjust_boxes(coarse_boxes,models);

    if (params.detect_min_scene_os > 0.0)
      os = getosmatrix_bb(boxes,[1 1 size(I,2) size(I,1)]);
      goods = find(os >= params.detect_min_scene_os);
      boxes = boxes(goods,:);
      coarse_boxes = coarse_boxes(goods,:);
    end
    
    if display == 1       
      %extract detection box vectors from the localization results
      saveboxes = boxes;
      if size(boxes,1)>=1
        boxes(:,5) = 1:size(boxes,1);
      end
      
      % if exist('M','var') && length(M)>0
      %   boxes = calibrate_boxes(boxes, M.betas);
      % end

      if numel(boxes)>0
        [aa,bb] = sort(boxes(:,end),'descend');
        boxes = boxes(bb,:);
      end
 
      %already nmsed (but not for LRs)
      boxes = nms_within_exemplars(boxes,.5);


      %% ONLY SHOW TOP 5 detections or fewer
      boxes = boxes(1:min(size(boxes,1),5),:);
      
      if 1% size(boxes,1) >=1
        figure(53)
        clf
        % stuff.filer = '';               
        % exemplar_overlay = exemplar_inpaint(boxes(1,:), ...
        %                                     models{boxes(1,6)}, ...
        %                                     stuff);

        % show_hits_figure_iccv(models,boxes,I,I,exemplar_overlay,I);

        show_hits_figure(models, boxes, I);
        drawnow
        %pause(.1)
%       else
%         figure(1)
%         clf
%         imagesc(I)
%         drawnow
%         fprintf(1,'No detections in this Image\n');
%         pause(.1)
      end
      boxes = saveboxes;
    end

    extras = [];
    res{j}.coarse_boxes = coarse_boxes;
    res{j}.bboxes = boxes;

    res{j}.index = index;
    res{j}.extras = extras;
    res{j}.imbb = [1 1 size(I,2) size(I,1)];
    res{j}.curid = curid;

    %%%NOTE: the gt-function is well-defined for VOC-exemplars
    if isfield(params,'gt_function') && ...
          ~isempty(params.gt_function)
      res{j}.extras = params.gt_function(params.dataset_params, Iname, res{j}.bboxes);
    end 
  end


  counter = counter + L;

  if display == 1
    continue
  end
  
  % save results into file and remove lock file
  
  if save_files == 1
    save(filer,'res');
    try
      rmdir(filerlock);
    catch
      fprintf(1,'Directory %s already gone\n',filerlock);
    end
  else
    allfiles{i} = res;
  end
end

if save_files == 0
  grid = cellfun2(@(x)x{1},allfiles);
  return;
end

[allfiles] = sort(allfiles);
grid = esvm_load_result_grid(params.dataset_params, models, ...
                             setname, ...
                             allfiles);

