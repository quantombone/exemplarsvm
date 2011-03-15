function exemplar_initialize(cls)
%% Initialize script which writes out initial model files for all
%% exemplars of a single category from PASCAL VOC trainval set
%% Script is parallelizable
%% Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;

%Store exemplars for this class
if ~exist('cls','var')
  cls = 'train';
  %cls = 'cow';
  %cls = 'tvmonitor';
  %cls = 'bicycle';
  %cls = 'chair';
  cls = 'dog';
  cls = 'car';
  cls = 'motorbike';
  cls = 'diningtable';
  cls = 'horse';
  cls = 'bus';
  cls = 'aeroplane';
  cls = 'bottle';
  cls = 'bird';
  cls = 'sofa';
  cls = 'pottedplant';
  %cls = 'person';
  %these are the voc2010 classes we processed
  
  cls = 'cow';
  cls = 'bus';
  cls = 'motorbike';
  cls = 'sofa';
  cls = 'diningtable';
  cls = 'sheep';
  cls = 'train';
  cls = 'cow';
end

if ismember(cls,{'all'})
  classes = VOCopts.classes;
  
  r = randperm(length(classes));
  for i = 1:length(classes)
    initialize_voc_exemplars(classes{r(i)});
  end
  return;
end

results_directory = ...
    sprintf('%s/exemplars2/',VOCopts.localdir);

fprintf(1,'Writing Exemplars of class %s to directory %s\n',cls,results_directory);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'trainval'),...
                  '%s %d');
ids = ids(gt==1);

myRandomize;
rrr = randperm(length(ids));
ids = ids(rrr);

%HACK
%ids = cell(1,1);
%ids = {'003370'};

for i = 1:length(ids)
  curid = ids{i};
  %curid = '004354';
  %curid = '2008_006368';
  %curid = '2008_006158';
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
  Ibase = imread(sprintf(VOCopts.imgpath,curid));
  Ibase = im2double(Ibase);
  
  for objectid = 1:length(recs.objects)
    %objectid = 1;
    
    
    %skip difficult objects, and objects not of target class
    if (recs.objects(objectid).difficult==1) | ...
          ~ismember({recs.objects(objectid).class},{cls})
      continue
    end
      
    fprintf(1,'.');
    
    %figure(1)
    %clf
    %imagesc(Ibase)
    %plot_bbox(recs.objects(objectid).bbox)
    %title(sprintf('i=%d objectid=%d',i,objectid));
    %pause
    %continue

    filer = sprintf('%s/%s.%d.%s.mat',results_directory,curid,objectid,cls);
    filerlock = [filer '.lock'];
    if fileexists(filer) || (mymkdir_dist(filerlock)==0)
      continue
    end
        
    bbox = recs.objects(objectid).bbox;
    gt_box = bbox;
    I = Ibase;

    %Get the hog features (+wiggles) from the ground-truth bounding box
    clear model;
    
    I2 = zeros(size(I,1),size(I,2));
    %keyboard
    %fprintf(1,'Growing region...');
    for expandloop = 1:10000
      % Get initial dimensions
      w = bbox(3)-bbox(1)+1;
      h = bbox(4)-bbox(2)+1;
      
      if h > w*4 || w < 50
        %% make wider
        bbox(3) = bbox(3) + 1;
        bbox(1) = bbox(1) - 1;
      elseif w > h*4 || h < 50
        %make taller
        bbox(4) = bbox(4) + 1;
        bbox(2) = bbox(2) - 1;
      else
        break;
      end
      
      bbox([1 3]) = cap_range(bbox([1 3]), 1, size(I,2));
      bbox([2 4]) = cap_range(bbox([2 4]), 1, size(I,1));      
    end
    fprintf(1,' Done to %d %d w/h=%.3f h/w=%.3f\n',w,h,w/h,h/w);
 
    %chunk = I2(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I2(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;%rand(size(chunk));
    model.params.sbin = 8;
        
    ARTPAD = 0; %120;
    %I_noise_pad = pad_image(I2,ARTPAD);
    I_real_pad = pad_image(I,ARTPAD);
    
    fprintf(1,'doing pyramids...');
    
    %f = featpyramid2(I_noise_pad,model.params.sbin, 10);
    [f_real,scales] = featpyramid2(I_real_pad,model.params.sbin, 10);
    fprintf(1,'done\n');
    clear masker
    clear sizer
    for a = 1:length(f_real)
      masker{a} = max(0.0,min(1.0,imresize(I2,[size(f_real{a},1) size(f_real{a}, ...
                                                      2)])));
      [tmpval,ind] = max(masker{a}(:));
      masker{a} = (masker{a}>.1);
      if sum(masker{a}(:))==0
        [aa,bb] = ind2sub(size(masker{a}),ind);
        masker{a}(aa,bb) = 1;
      end
      [uu,vv] = find(masker{a});
      sizer(a,:) = [range(uu)+1 range(vv)+1];
    end
    
    GOAL_NCELLS = 100;
    fprintf(1,'GOAL_NCELLS=%d\n',GOAL_NCELLS);
    ncells = prod(sizer,2);
    [aa,targetlvl] = min(abs(ncells-GOAL_NCELLS));
     
    [uu,vv] = find(masker{targetlvl});
    curfeats = f_real{targetlvl}(min(uu):max(uu),min(vv):max(vv),: ...
                                 );

    
    model.w = curfeats - mean(curfeats(:));
    model.b = 0;
    mmm{1}.model = model;
    mmm{1}.model.hg_size = size(model.w);
    
    localizeparams.thresh = -100.0;
    localizeparams.TOPK = 1;
    localizeparams.lpo = 10;
    localizeparams.SAVE_SVS = 1;
    
    %%model.target_id.level = targetlvl;
    %model.target_id.scale = scales(targetlvl);
    %model.target_id.offset = [min(uu) min(vv)];
    

    if 0
      for zzz = 1:4
        [rs,t] = localizemeHOG(I,mmm,localizeparams);
        
        f = reshape(rs.support_grid{1}{1},size(mmm{1}.model.w));
        sf = sum(f.^2,3);
        nbads = sum(sf(:)==0)
        if nbads == 0
          break;
        end
        [u,v] = find(sf>0);
        curw = f(min(u):max(u),min(v):max(v),:);
        curw = curw - mean(curw(:));
        mmm{1}.model.w = curw;
        mmm{1}.model.hg_size = size(curw);  
      end
      
      model.target_id = rs.id_grid{1}{1};
      model.hg_size = mmm{1}.model.hg_size;
    else  
      [rs,t] = localizemeHOG(I,mmm,localizeparams);
      model.target_id = rs.id_grid{1}{1};
      % figure(1)
      % clf
      % imagesc(I)
      % plot_bbox(model.target_id.bb)
      % axis image
      % drawnow
      model.hg_size = size(curfeats);
    end
    
    NWIGGLES = 100;

    %Get wiggles
    xxx = replica_hits(I, model.params.sbin, model.target_id, ...
                       model.hg_size, NWIGGLES);

    %% GET self feature + 100 wiggles
    model.x = xxx;
    model.w = reshape(mean(model.x,2), model.hg_size);
    model.w = model.w - mean(model.w(:));
    model.b = -100;
    
    model.coarse_box = rs.id_grid{1}{1}.bb;
    model.nsv = zeros(prod(model.hg_size),0);
    model.svids = [];
    
    m.curid = curid;
    m.objectid = objectid;
    m.cls = cls;
  
    %if class is '?' then we are category free and we mine all
    %images except target id
    %m.cls = '?';
    
    m.gt_box = gt_box;
    m.model = model;
    m.sizeI = size(I);
    
    fprintf(1,'Final OS is %.3f\n',getosmatrix_bb(m.gt_box, m.model.coarse_box));

    save(filer,'m');
    if exist(filerlock,'dir')
      rmdir(filerlock);
    end

    figure(1)
    clf
    imagesc(Ibase)
    plot_bbox(m.model.coarse_box)
    axis image
    drawnow
  end  
end
