function resser = initialize_all_positives(cls)
%% Initialize all positive examples with their wiggles
%% Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;

%Store exemplars for this class
if ~exist('cls','var')
  %cls = 'all';
  %Cls = 'train';
  cls = 'cow';
end

results_directory = ...
    sprintf('%s/exemplars/',VOCopts.localdir);

fprintf(1,'Writing Exemplars of class %s to directory %s\n',cls,results_directory);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,'trainval'),...
                  '%s %d');
ids = ids(gt==1);

if nargout==0
  myRandomize;
  rrr = randperm(length(ids));
  ids = ids(rrr);
end

resser = cell(0,1);
for i = 1:length(ids)
  curid = ids{i};
  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
  Ibase = imread(sprintf(VOCopts.imgpath,curid));
  Ibase = im2double(Ibase);
  
  for objectid = 1:length(recs.objects)
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

    filer = sprintf('%s/allp.%s.%d.%s.mat',results_directory,...
                    curid,objectid,cls);
    filerlock = [filer '.lock'];
    if fileexists(filer)
      resser{end+1} = load(filer);
      resser{end} = resser{end}.model;
      if length(resser{end}.curw)==0
        resser = resser(1:end-1);
      end
      continue
    end
    if (mymkdir_dist(filerlock)==0)
      continue
    end
        
    bbox = recs.objects(objectid).bbox;
    I = Ibase;

    %Get the hog features (+wiggles) from the ground-truth bounding box
    clear model;
    
    model.params.sbin = 8;
    model.params.MAX_CELL_DIM = 12;
    model.params.MIN_CELL_DIM = 3;
    model.params.SVMC = .01;    
    
    %mask = logical(zeros(size(I,1),size(I,2)));
    %mask(bbox(2):bbox(4),bbox(1):bbox(3))=1;
    
    %[model.x, model.hg_size, model.coarse_box, newmasks] = ...
    %    hog_from_bbox(I, bbox, model.params, mask);
    
    %cellfun(@(x)size(x,1),x) - round((round(size(I,1)*scales))/8)
    
    save_bbox = bbox;
    Isave = I;
    

    % for cx = -10:2:10
    %   for cy = -10:2:10
        
    bbox_pad = bbox;
    
    % W = bbox(4) - bbox(2)+1;
    % H = bbox(3) - bbox(1)+1;
    % FRAC = .2;
    % bbox_pad(1) = round(bbox_pad(1) - H*FRAC);
    % bbox_pad(2) = round(bbox_pad(2) - W*FRAC);
    % bbox_pad(3) = round(bbox_pad(3) + H*FRAC);
    % bbox_pad(4) = round(bbox_pad(4) + W*FRAC);
    % bbox_pad([1 3]) = cap_range(bbox_pad([1 3]),1,size(I,2));
    % bbox_pad([2 4]) = cap_range(bbox_pad([2 4]),1,size(I,1));
    
    masker = zeros(size(I,1),size(I,2),1);
    masker(bbox_pad(2):bbox_pad(4),bbox_pad(1):bbox_pad(3)) = 1;    
    masker = repmat(masker,[1 1 3]);
    m2 = find(masker);
    masker(m2) = rand(size(m2));
    
    rmasker = zeros(size(I,1),size(I,2),1);
    rmasker(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;    
    rmasker = repmat(rmasker,[1 1 3]);
    m2 = find(rmasker);
    rmasker(m2) = rand(size(m2));
 
    LPO = 10;
    
    [x,scales] = featpyramid2(I, model.params.sbin, LPO);
    [m,scales] = featpyramid2(masker, model.params.sbin, LPO);
    [rm,scales] = featpyramid2(rmasker, model.params.sbin, LPO);
    
    rrr = cellfun(@(x)sum(abs(x(:))),rm);
    fbad = find(rrr==0);
    if length(fbad)>0
      fbad = fbad(1)-1;
      x = x(1:fbad);
      m = m(1:fbad);
      rm = rm(1:fbad);
      scales = scales(1:fbad);
    end

    % figure(1)
    % clf
    % imagesc(I)
    % plot_bbox(bbox)
    % plot_bbox(bbox_pad,'',[1 0 0])
    
    clear curw curb curm
    for iii = 1:length(scales)
      hits = find(sum(m{iii}.^2,3)>0);
      [u,v] = ind2sub([size(m{iii},1) size(m{iii},2)],hits);
      minu = min(u);
      maxu = max(u);
      minv = min(v);
      maxv = max(v);
      curw{iii} = x{iii}(minu:maxu,minv:maxv,:);
      cmasker = sum(rm{iii}.^2,3)>0;
      curm{iii} = cmasker(minu:maxu,minv:maxv);
      curb{iii} = zeros(1,4);

      curb{iii}([2 4]) = [minu-1 maxu]*model.params.sbin/scales(iii);
      curb{iii}([1 3]) = [minv-1 maxv]*model.params.sbin/scales(iii);
      curb{iii} = curb{iii} + 1;
    end

    
    s1 = cellfun(@(x)size(x,1), curw);
    s2 = cellfun(@(x)size(x,2), curw);
    maxs = max(s1,s2);
    mins = min(s1,s2);
    
    %select a reasonable subset
    goods = find(mins <= 15 & maxs >= 7);
    curw = curw(goods);
    curb = curb(goods);
    curm = curm(goods);
    model.curw = curw;
    model.curb = curb;
    model.curm = curm;
    
    model.curid = curid;
    model.objectid = objectid;
    model.cls = cls;
      
    model.gt_box = bbox;
    model.gt_box_padded = bbox_pad;
   

    save(filer,'model');
    if exist(filerlock,'dir')
      rmdir(filerlock);
    end

  end  
end
