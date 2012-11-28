function grid = esvm_detect_imageset(imageset, model, ...
                                     params, setname)
% Apply a model (ExemplarSVMs, Mixture Models, or DalalTriggs) to a
% set of images.
%
% imageset: a (virtual) set of images, such that
%   toI(imageset{i}) returns an image
% model: A model
% params(optional): detection parameters
% setname(optional): a name of the set, which lets us cache results
%   on disk
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm
%fprintf(1,' \n---Detect imageset running now\n');

if ~exist('params','var')
  params = esvm_get_default_params;
end

if ~exist('setname','var') || length(setname)==0
  params.detect_images_per_chunk = 1;
  setname = '';
end

if length(params.localdir)>0 && length(setname)>0
  save_files = 1;
else  
  save_files = 0;
end

if isempty(imageset)
  grid = {};
  return;
end

if save_files == 1
  
  model_name = '';
  if length(model)>=1 && isfield(model{1},'model_name') && ...
        isstr(model{1}.model_name)
    model_name = model{1}.model_name;
  end
  final_file = sprintf('%s/detections/%s-%s.mat',...
                       params.localdir,setname, ...
                       model_name);

  if fileexists(final_file)
    res = load(final_file);
    grid = res.grid;
    return;
  end
end


if save_files == 1
  baser = sprintf('%s/detections/%s-%s/',params.localdir,setname, ...
                  model_name);
else
  baser = '';
end

if (save_files==1) && (~exist(baser,'dir'))
  fprintf(1,'Making directory %s\n',baser);
  mkdir(baser);
end

%% Chunk the data into detect_images_per_chunk images per chunk so that we
%process several images, then write results for entire chunk

inds = do_partition(1:length(imageset),params.detect_images_per_chunk);

% randomize chunk orderings, when writing to disk
if length(params.localdir) > 0 && (params.display ~= 1)
  myRandomize;
  ordering = randperm(length(inds));
else
  ordering = 1:length(inds);
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
    if (fileexists(filer) || (mymkdir_dist(filerlock) == 0))
      continue
    end
  end
  res = cell(0,1);

  %clear Is;
  Is = imageset(inds{ordering(i)});
  L = length(inds{ordering(i)});

  for j = 1:L

    index = inds{ordering(i)}(j);
    %fprintf(1,' --image %05d/%05d:',counter+j,length(imageset));
    fprintf(1,'.');

    Iname = imageset{index};

    curid = '';
    % try
    %   hit = strfind(Iname,'JPEGImages/');
    %   curid = Iname((hit+11):end);
    %   hit = strfind(curid,'.');
    %   curid = curid(1:(hit(end)-1));      
    %   %[tmp,curid,tmp] = fileparts(Iname);
    % catch
    %   curid = '';
    % end
    

    I = toI(Is{j});
       
    starter = tic;
    %params.detect_save_features=1;
    rs = esvm_detect(I, model, params);

    if isfield(model.models{1},'A')
      x = cat(2,rs.xs{1}{:});
      x(end+1,:) = 1;
      b2 = model.models{1}.A*x;
      for qqq = 1:size(rs.bbs{1},1)
        curbb = rs.bbs{1}(qqq,:);
        xform = find_xform(model.models{1}.center,curbb);
        newbb = apply_xform(b2(:,qqq)',xform);
        rs.bbs{1}(qqq,1:4) = newbb(1:4);      
      end
    end
    
    coarse_boxes = cat(1,rs.bbs{:});
    if ~isempty(coarse_boxes)
      coarse_boxes(:,11) = index;
      scores = coarse_boxes(:,end);
    else
      scores = [];
    end
    [aa,bb] = max(scores);
    %fprintf(1,' %d w''s took %.3fsec, #windows=%05d, max=%.3f \n',...
    %        length(model.models),toc(starter),length(scores),aa);
    
    coarse_boxes(:,5) = 1:length(coarse_boxes(:,5));
    coarse_boxes = esvm_nms_within_exemplars(coarse_boxes);
    %goods = coarse_boxes(:,5);
    %coarse_boxes = coarse_boxes(goods,:);
    

    
    % Transfer GT boxes from model onto the detection windows
    %NOTE: this is very slow
    boxes = esvm_adjust_boxes(coarse_boxes, model);
    %boxes = coarse_boxes;
    
    

    if (params.detect_min_scene_os > 0.0)
      os = getosmatrix_bb(boxes,[1 1 size(I,2) size(I,1)]);
      goods = find(os >= params.detect_min_scene_os);
      boxes = boxes(goods,:);
      coarse_boxes = coarse_boxes(goods,:);
    end
    
    extras = [];


    %res = esvm_pool_exemplar_dets(boxes, model.models, M, ...
    %                          test_params);
    if isfield(model,'betas')
      boxes = esvm_calibrate_boxes(boxes, model.betas);
    end

    if isfield(model,'M') && isfield(model.M,'w')
      K = length(model.models);
      xraw = esvm_get_M_features(boxes, K, .5);
      boxes = esvm_apply_M(xraw,boxes,model.M);
    end
    
    coarse_boxes(:,end) = boxes(:,end);
    res{j}.coarse_boxes = coarse_boxes;
    res{j}.bboxes = boxes;

    res{j}.index = index;
    res{j}.extras = extras;
    res{j}.imbb = [1 1 size(I,2) size(I,1)];
    res{j}.curid = curid;

    if params.display_detections > 0

      figure(1)
      clf
      
      [alpha,beta] = max(boxes(:,end));
      bbtop = boxes(beta,:);
      
      if 1
        
        [bbtop2] = esvm_nms(bbtop,.8);

        %bbtop2 = bbtop;
        bbtop3 = cellfun2(@(x)x.bb,model.models(bbtop2(:,6)));
        bbtop3 = cat(1,bbtop3{:});
        bbtop3 = bbtop3(1,:);

        bbtop3 = model.models{bbtop2(1,6)}.bb(1,:);
        if bbtop2(1,7)==1
          bbtop3(7)=1;
        end
        %bbtop3 = bbtop;

        %axis image
        %axis off
        
        if isfield(model,'data_set')
          
          I2 = showTopDetections(model.data_set,esvm_adjust_boxes(bbtop3,model));
          
          factor = size(I,1)/size(I2,1);
          
          I = cat(2,I,max(0.0,min(1.0,imresize(I2,round(factor*[size(I2,1) size(I2,2)])))));%size(I2,1)/size(I,1)
                                                                                            %figure(2)
                                                                                            %imagesc(I2)
        end
      end

      imagesc(I)%pad_image(I,100))
      
        
      if size(boxes,1) > 0 && max(boxes(:,end))>=-1
        % plot_bbox(esvm_nms(clip_to_image(boxes,[1 1 size(I,2) ...
        %             size(I,1)]),.5))
        %

        %bbtop = esvm_adjust_boxes(bbtop, model);
        [~,tops] = sort(boxes(:,end),'descend');
        tops = boxes(tops(1:min(length(tops),10)),:);
        %plot_bbox(tops);
        bbtop(1,1:4) = bbtop(1,1:4);
        %bbtop = (clip_to_image(bbtop,[1 1 size(I,2) size(I,1)]));
        
        plot_bbox(bbtop,[model.models{bbtop(1,6)}.cls ' ' num2str(bbtop(1,end))],[1 0 0])
        title(num2str(boxes(beta,end)))
        
     
        % 
        % bbtop2 = bbtop2(1:min(size(bbtop2,1),3),:);
        % plot_bbox(bbtop2)


      end
      %axis off
      %axis image
      axis image
      axis off
      
      drawnow
      pause(params.display_detections)

    end
    
    % if params.write_top_detection == 1
    %   if size(boxes,1) > 0
    %     bb = round(boxes(1,:));
    %   else
    %     bb = [1 1 3 3];
    %   end
    %   % fid = fopen('/tmp/coords.txt','w');
    %   % fprintf(fid,'%d %d %d %d\n',bb(1),bb(2),bb(3),bb(4));
    %   % fclose(fid);
      
    %   % [aa,bb]=unix(' kill -SIGINT `ps aux | grep python | grep rectframe.py | head -1 | awk ''{print($2)}''`');
    %   % fprintf(1,'just sent signal 1\n');
    %   % pause(.1)



      
    % end

    
    %%%NOTE: the gt-function is well-defined for VOC-exemplars
    % if isfield(params,'gt_function') && ...
    %       ~isempty(params.gt_function)
    %   res{j}.extras = params.gt_function(params.dataset_params, ...
    %                                      Iname, res{j}.bboxes);
    % end 
  end

  counter = counter + L;
  
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
grid = esvm_load_result_grid(params, model, setname, allfiles);
