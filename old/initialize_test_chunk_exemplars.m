function initialize_test_exemplars(id)
%% Initialize script which writes out initial model files for all
%% exemplars of a single category from PASCAL VOC trainval set
%% Script is parallelizable
%% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('id','var')
  id = 1;
end

VOCinit;

results_directory = ...
    sprintf('%s/test_chunks/',VOCopts.localdir);

fprintf(1,'Writing Exemplars of id %d to directory %s\n',id,...
        results_directory);

if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

[ids,gt] = textread(sprintf(VOCopts.imgsetpath,'test'),...
                  '%s %d');

curid = ids{id};

%curid = '2008_006368';
%curid = '2008_006158';
%recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
Ibase = imread(sprintf(VOCopts.imgpath,curid));
Ibase = im2double(Ibase);


fprintf(1,'.');

%figure(1)
%clf
%imagesc(Ibase)
%plot_bbox(recs.objects(objectid).bbox)
%title(sprintf('i=%d objectid=%d',i,objectid));
%pause
%continue

filer = sprintf('%s/%s.mat',results_directory,curid);
filerlock = [filer '.lock'];
%if fileexists(filer) || (mymkdir_dist(filerlock)==0)
%  return;
%end

bboxes = generate_bbs(Ibase);
ms = cell(0,1);
for qqq = 1:size(bboxes,1)
bbox = bboxes(qqq,:);

gt_box = bbox;
I = Ibase;

%Get the hog features (+wiggles) from the ground-truth bounding box
clear model;

I2 = zeros(size(I));
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

    chunk = I2(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I2(bbox(2):bbox(4),bbox(1):bbox(3),:) = rand(size(chunk));
    model.params.sbin = 8;
    
    
    ARTPAD = 120;
    I_noise_pad = pad_image(I2,ARTPAD);
    I_real_pad = pad_image(I,ARTPAD);
    
    f = featpyramid2(I_noise_pad,model.params.sbin, 10);
    f_real = featpyramid2(I_real_pad,model.params.sbin, 10);
    
    clear masker
    clear sizer
    for a = 1:length(f)
      masker{a} = (sum(f{a}.^2,3)>0);
      [uu,vv] = find(masker{a});
      sizer(a,:) = [range(uu)+1 range(vv)+1];
    end
    
    GOAL_NCELLS = 100;
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
    
    for zzz = 1:4
      [rs,t] = localizemeHOG(I,mmm,localizeparams);
    
      f = reshape(rs.support_grid{1}{1},size(mmm{1}.model.w));
      sf = sum(f.^2,3);
      nbads = sum(sf(:)==0);
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
    
    
    NWIGGLES = 100;
    
    %Get wiggles
    xxx = replica_hits(I, model.params.sbin, model.target_id, ...
                       model.hg_size, NWIGGLES);

    %% GET self feature + 100 wiggles
    model.x = xxx;
    model.w = reshape(mean(model.x,2), model.hg_size);
    model.w = model.w - mean(model.w(:));
    model.b = -100;
    
    mx = mean(xxx,2);
    mx = reshape(mx,model.hg_size);
    sums = sum(mx.^2,3);
   
    model.mask = (sum(mx.^2,3)>0);
    
    model.coarse_box = rs.id_grid{1}{1}.bb;
    model.nsv = zeros(prod(model.hg_size),0);
    model.svids = [];
    
    m.curid = curid;
      
    m.gt_box = gt_box;
    m.model = model;
    figure(1)
    clf
    imagesc(HOGpicture(model.w))
    drawnow
    pause(.1)
  
    fprintf(1,'Final OS is %.3f\n',getosmatrix_bb(m.gt_box, m.model.coarse_box));
    ms{end+1} = m;
end


save(filer,'ms');
if exist(filerlock,'dir')
  rmdir(filerlock);
end

function bbs = generate_bbs(I)
scaler = 100/max(size(I));
I2 = imresize(I,scaler);
I2 = min(1.0,max(0.0,I2));
res=ncut_multiscale(I2,5);
res = myremove_small_segments(res,100,I2);

ures = unique(res);
bbs = zeros(length(ures),4);
for i = 1:length(ures)
  [u,v] = find(res==ures(i));
  factor1 = .15*(max(v)-min(v));
  factor2 = .15*(max(u)-min(u));
  bbs(i,:) = round([min(v)-factor1 min(u)-factor2 max(v)+factor1 max(u)+factor2]/scaler);
end

bbs = clip_to_image(bbs,[1 1 size(I,2) size(I,1)]);
