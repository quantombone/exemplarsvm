function [models2] = initialize_some_exemplars(Is)
% Initialize script which given a cell array of images, finds a
% region in each image (by user interaction)
% fg:       a cell array of images
% [models]: a cell array of models used to automatically select
%           input window
% models2:  resulting cell array of model files
% Tomasz Malisiewicz (tomasz@cmu.edu)

%results_directory = ...
%    sprintf('/projects/ddip/fgdata/exemplars/');

%fprintf(1,'Writing Exemplars to directory %s\n',results_directory);

%if ~exist(results_directory,'dir')
%  fprintf(1,'Making directory %s\n',results_directory);
%  mkdir(results_directory);
%end

N_PER_FRAME = 1;

VOCinit;

results_directory = ...
    sprintf('%s/exemplars/',VOCopts.localdir);

fprintf(1,'Writing Exemplars to directory %s\n',results_directory);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

do_pascal = 0;
if exist('Is','var')
  %%use provided images
  LEN = length(Is);
else
    do_pascal = 1;
    %% Load ids of all images in trainval that contain cls
    [ids,gt] = textread(sprintf(VOCopts.imgsetpath,'trainval'),...
                        '%s %d');
    LEN = length(ids);
end

models2 = cell(0,1);

for i = 1:min(10,LEN)
  

  if do_pascal == 1
      curid = ids{i};
      Ibase = imread(sprintf(VOCopts.imgpath,ids{i}));
  else
      curid = sprintf('my_%d',i);
      Ibase = Is{i};
  end
  Ibase = im2double(Ibase);

  for objectid = 1:N_PER_FRAME
    
    filer = sprintf('%s/%s.%d.mat',results_directory,curid,objectid);
    filerlock = [filer '.lock'];
    if fileexists(filer) || (mymkdir_dist(filerlock)==0)
      %fprintf(1,'HACK skipping locks\n');
      continue
    end
    
    if ~exist('models','var') | ~exist('bbox','var')
      figure(1)
      clf
      imagesc(Ibase)
      title(sprintf('enter box coords %d',objectid));
      
      % try to automatically segment out
      % outsiderow = squeeze(Ibase(1,1,:));
      % Irow = reshape(Ibase,[],3)';
      % ddd = distSqr_fast(Irow,outsiderow);
      % ddd = reshape(ddd',size(Ibase,1),size(Ibase,2));
      % seg = bwmorph((ddd>0),'dilate',2);
      % [u,v] = find(seg);
      % x(1) = min(v);
      % x(2) = max(v);
      % y(1) = min(u);
      % y(2) = max(u);
      
      [x,y] = ginput(2);  
      bbox = round([x(1) y(1) x(2) y(2)]);
      plot_bbox(bbox)      
    elseif exist('models','var') & length(models)>0
      %Find single top detection
      [rs] = localizemeHOG(Ibase,ws,bs,-2.00,1,10);
      bbox = extract_bbs_from_rs(rs,Ibase,models);
    end
    I = Ibase;
    
        %Get the hog features (+wiggles) from the ground-truth bounding box
    clear model;
    [model.x, model.hg_size, model.coarse_box] = hog_from_bbox(I,bbox);
    
    model.hg_size = [model.hg_size 31];
    model.w = reshape(mean(model.x,2),model.hg_size);
    model.b = 0;
    model.nsv = zeros(prod(model.hg_size),0);
    model.svids = [];
    model.mine_ids = [];
  
    m.curid = curid;
    m.objectid = objectid;
    m.cls = 'selection';
    m.I = I;
  
    %Set the name of this exemplar type
    m.models_name = 'cvpr11_model';
    
    %Set up the negative set for this exemplars
    %CVPR2011 paper used all train images excluding category images
    %m.bg = sprintf('get_pascal_bg(''train'')');%,''-%s'')',m.cls);
    m.bg = sprintf('get_movie_bg');
    %if class is '?' then we are category free and we mine all
    %images except target id
    %m.cls = '?';
    
    m.gt_box = bbox;
    m.model = model;

    models2{end+1} = m;
    
    if 0 %%nargin==1
        scaler = [.25 .4 .6 .8];
        for iii = 1:length(scaler)
            
            c1 = round(mean(m.gt_box([2 4])));
            c2 = round(mean(m.gt_box([1 3])));
            
            I2 = m.I;
            I2(c1:m.gt_box(4),m.gt_box(1):m.gt_box(3),:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);
            
            I2 = m.I;
            I2(m.gt_box(1):c1,m.gt_box(1):m.gt_box(3),:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);
            
            I2 = m.I;
            I2(m.gt_box(1):m.gt_box(3),c2:m.gt_box(3),:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);
            
            I2 = m.I;
            I2(m.gt_box(1):m.gt_box(3),m.gt_box(1):c2,:)=0;
            I2 = max(0.0,min(1.0,imresize(imresize(I2,scaler(iii)), ...
                                          [size(m.I,1) size(m.I,2)])));
            m2 = initialize_exemplars_fg({I2},[],bbox);
            m.coarse_box = cat(1,m.coarse_box,m2.coarse_box);
            m.model.x = cat(2,m.model.x,m2.model.x);

        end
    end

    save(filer,'m');
    rmdir(filerlock);
  end  
end
