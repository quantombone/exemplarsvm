function allfiles = apply_all_exemplars(dataset_params,models,fg,setname,M)
% Apply a set of models (raw exemplars, trained exemplars, dalals,
% poselets, components, etc) to a set of images.  Script can be ran in
% parallel with no arguments.  After running script, use
% grid=load_result_grid(models) to load results.
%
% models: Input cell array of models (try models=load_all_models)
% M: The boosting Matrix (optional)
% curset: The PASCAL VOC image set to apply the exemplars to
%   ('trainval' or 'test')
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%Save results every NIMS_PER_CHUNK images

if length(fg) == 0
  allfiles = {};
  return;
end

NIMS_PER_CHUNK = dataset_params.NIMS_PER_CHUNK;

%Only allow display to be enabled on a machine with X
display = dataset_params.display;

if display == 1
  fprintf(1,'DISPLAY ENABLED, NOT SAVING RESULTS!\n');
end

if ~isfield(dataset_params,'params')
  params = get_default_mining_params;
else
  params = dataset_params.params;
end

%if strcmp(models{1}.models_name,'dalal')
%  params.TOPK = 100;
%  params.thresher = -2.5;
%end

setname = [setname '.' models{1}.cls];

baser = sprintf('%s/applied/%s-%s/',dataset_params.localdir,setname, ...
                models{1}.models_name);

if ~exist(baser,'dir')
  fprintf(1,'Making directory %s\n',baser);
  mkdir(baser);
end

%% Chunk the data into NIMS_PER_CHUNK images per chunk so that we
%process several images, then write results for entire chunk

inds = do_partition(1:length(fg),NIMS_PER_CHUNK);

% randomize chunk orderings
myRandomize;
ordering = randperm(length(inds));
if display == 1
  ordering = 1:length(ordering);
end

%[v,host_string]=unix('hostname');

allfiles = cell(length(ordering), 1);

for i = 1:length(ordering)

  ind1 = inds{ordering(i)}(1);
  ind2 = inds{ordering(i)}(end);
  filer = sprintf('%s/result_%05d-%05d.mat',baser,ind1,ind2);
  allfiles{i} = filer;
  filerlock = [filer '.lock'];


  if fileexists(filer) || (mymkdir_dist(filerlock) == 0)
    continue
  end

  
  res = cell(0,1);

  %% pre-load all images in a chunk
  fprintf(1,'Preloading %d images\n',length(inds{ordering(i)}));
  clear Is;
  for j = 1:length(inds{ordering(i)})
    Is{j} = convert_to_I(fg{inds{ordering(i)}(j)});
  end
  
  for j = 1:length(inds{ordering(i)})

    index = inds{ordering(i)}(j);
    fprintf(1,'   ---image %d\n',index);
    Iname = fg{index};
    [tmp,curid,tmp] = fileparts(Iname);
    
    I = Is{j};
       
    starter = tic;
    [rs,t] = localizemeHOG(I, models, params);
    
    for q = 1:length(rs.bbs)
      if ~isempty(rs.bbs{q})
        rs.bbs{q}(:,11) = index;
        if length(rs.bbs{q}(1,:))~=12
          error('BUG: Invalid length bb');
        end
      end
    end
        
    coarse_boxes = cat(1,rs.bbs{:});
    if ~isempty(coarse_boxes)
      scores = coarse_boxes(:,end);
    else
      scores = [];
    end
    [aa,bb] = max(scores);
    fprintf(1,' took %.3fsec, maxhit=%.3f, #hits=%d\n',...
            toc(starter),aa,length(scores));
    
    % Transfer GT boxes from models onto the detection windows
    boxes = adjust_boxes(coarse_boxes,models);
    
    if (params.MIN_SCENE_OS > 0.0)
      os = getosmatrix_bb(boxes,[1 1 size(I,2) size(I,1)]);
      goods = find(os >= params.MIN_SCENE_OS);
      boxes = boxes(goods,:);
      coarse_boxes = coarse_boxes(goods,:);
    end


    if display == 1       
      %extract detection box vectors from the localization results
      saveboxes = boxes;
      if size(boxes,1)>=1
        boxes(:,5) = 1:size(boxes,1);
      end
      
      if exist('M','var') && length(M)>0
        boxes = calibrate_boxes(boxes, M.betas);
      end

      if numel(boxes)>0
        [aa,bb] = sort(boxes(:,end),'descend');
        boxes = boxes(bb,:);
      end
 
      %already nmsed (but not for LRs)
      boxes = nms_within_exemplars(boxes,.5);

      %% ONLY SHOW TOP 5 detections or fewer
      boxes = boxes(1:min(size(boxes,1),8),:);
      
      if size(boxes,1) >=1
        figure(1)
        clf
        % stuff.filer = '';               
        % exemplar_overlay = exemplar_inpaint(boxes(1,:), ...
        %                                     models{boxes(1,6)}, ...
        %                                     stuff);

        % show_hits_figure_iccv(models,boxes,I,I,exemplar_overlay,I);
        show_hits_figure(models, boxes, I);
        drawnow
        pause(.1)
      else
        figure(1)
        clf
        imagesc(I)
        drawnow
        fprintf(1,'No detections in this Image\n');
        pause(.1)
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
    if isfield(params,'gt_function') && ~isempty(params.gt_function)
      res{j}.extras = params.gt_function(dataset_params, Iname, res{j}.bboxes);
    end 
  end
  
  % save results into file and remove lock file
  save(filer,'res');
  try
    rmdir(filerlock);
  catch
    fprintf(1,'Directory %s already gone\n',filerlock);
  end
  
end

[allfiles,bb] = sort(allfiles);
