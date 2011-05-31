function allbbs = show_top_dets(models, grid, target_directory, finalstruct)
%% Show the top detections for the exemplar model inside models,
%% where grid is the set of detections, target_directory is 'test',
%% and finalstruct (which contains final boxes) is obtained from the
%% evaluate_pascal_voc_grid script
%% Tomasz Malisiewicz (tomasz@cmu.edu)

VOCinit;
%finalstruct.basedir = '/nfs/baikal/tmalisie/labelme400/www/voc/iccv11/VOC2007/newones/';

allbbs = cell(0,1);
final_boxes = finalstruct.unclipped_boxes;
final_maxos = finalstruct.final_maxos;

%if enabled we show images
saveimages = 0;

%% prune grid to contain only images from target_directory
[cur_set, gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),['%s' ...
                    ' %d']);

if 0
  fprintf(1,'HACK: choosing only the subset which contains true positives\n');
  %% prune grid to contain only images from target_directory
  [cur_set, gt] = textread(sprintf(VOCopts.clsimgsetpath,models{1}.cls,target_directory),['%s' ...
                    ' %d']);
  cur_set = cur_set(gt==1);
end

gridids = cellfun2(@(x)x.curid,grid);
goods = find(ismember(gridids,cur_set));
grid = grid(goods);

imids = cell(1,length(final_boxes));
for i = 1:length(final_boxes)
  imids{i} = [];
  if size(final_boxes{i},1) > 0
    imids{i} = i * ones(size(final_boxes{i},1),1);
  end
end

bbs = cat(1,final_boxes{:});
imids = cat(1,imids{:});
moses = [final_maxos{:}];
[aa,bb] = sort(bbs(:,end),'descend');

%% only good ones now!
%bb = bb(moses(bb)>.5);

if 0
  fprintf(1,'Only segmentation ones here\n');
  %% get ones with segmentation only
  ids = cellfun2(@(x)x.curid,models(bbs(bb,6)));
  VOCinit
  has_seg = cellfun(@(x)fileexists(sprintf('%s/%s/SegmentationObject/%s.png',...
                                           VOCopts.datadir, ...
                                           VOCopts.dataset,x)),ids);
  bb = bb(find(has_seg));
end

if 0
  %% get ones with 3d model only
  ids = cellfun2(@(x)sprintf('%s.%d.png',x.curid,x.objectid),models(bbs(bb,6)));
  VOCinit
  
  extradir = '/nfs/hn22/tmalisie/ddip/renderings/';
  has_3d = cellfun(@(x)fileexists(sprintf('%s/%s.%d.chair-3d.png', ...
                                          extradir,...
                                          x.curid, ...
                                          x.objectid)),models(bbs(bb,6)));

  bb = bb(find(has_3d));
end

counter = 1;
maxk = 100;
KKK = [1 1];

counter = 1;

for k = 1:maxk
  k
  %figure(1)
  %clf
  for i = 1:prod(KKK)
    if counter > length(bb)
      break;
    end
    curb = bb(counter);
    curid = grid{imids(curb)}.curid;

    I = im2double(imread(sprintf(VOCopts.imgpath,curid)));
    gtrecs = PASreadrecord(sprintf(VOCopts.annopath,curid));
    businds = find(ismember({gtrecs.objects.class},{'bus'}) & ~[gtrecs.objects.difficult]);
    gtbbs = cat(1,gtrecs.objects.bbox);
    gtbbs = gtbbs(businds,:);

    bbox = bbs(curb,:);

    TARGET_BUS = -1;
    if length(businds) > 0
      [alpha,beta] = max(getosmatrix_bb(gtbbs,bbox));
      TARGET_BUS = businds(beta);
    end
        
    if length(moses) > 0
      stuff.os = moses(curb);
    else
      stuff.os = 0;
    end
    stuff.score = bbs(curb,end);
    stuff.curid = curid;
    stuff.rank = counter;
   
    extra = '';
    
    if ismember(models{1}.models_name,{'dalal'})
      extra='-dalal';
    end

    estring = '';
    if exist('betas','var')
      estring = 'withbetas';
    end
    
    stuff.filer = sprintf(['/nfs/baikal/tmalisie/labelme400/www/voc/'...
                    'segxfer/%s%s-on-%s-%s-%d.eps'],...
                          models{1}.cls,extra,target_directory,estring,k);
    ccc = bbox(6);
    
    target_image_id = imids(bb(counter));
    target_cluster_id = bbs(bb(counter),5);
        
    allbb = finalstruct.raw_boxes{target_image_id};
    osmat = getosmatrix_bb(allbb,bbs(bb(counter),:));
    goods = find(osmat>.5);
    allbb = allbb(goods,:);
    [alpha,beta] = sort(allbb(:,end),'descend');
    allbb = allbb(beta,:);
    
    sumI = I*0;
    countI = zeros(size(I,1),size(I,2),1);
    ooo = cell(0,1);
    
    mean0 = mean(allbb,1);
    curoses = getosmatrix_bb(mean0(1,:),allbb);
    
    for zzz = 1:min(1,size(allbb,1))

      exemplar_overlay = exemplar_inpaint(allbb(zzz,:), ...
                                          models{allbb(zzz,6)}, ...
                                          stuff);
      
      ooo{end+1} = exemplar_overlay;

      sumI = sumI + curoses(zzz)*allbb(zzz,end)*exemplar_overlay.segI;
      countI = countI + ...
               curoses(zzz)*allbb(zzz,end);                                       
    end
    sumI = sumI ./ repmat(countI,[1 1 3]);
    


    if TARGET_BUS > -1
      %% load GT for test 
      %gtfiles = dir(sprintf(['/nfs/baikal/tmalisie/buslabeling/' ...
      %                '%s*mat'],curid));
      gtmat = load(sprintf(['/nfs/baikal/tmalisie/finalbuslabeling/' ...
                    '%s.%d.mat'],curid,TARGET_BUS));

      %   clear gtmat
      % for i = 1:length(gtfiles)      
      %   gtmat{i} = load(sprintf('/nfs/baikal/tmalisie/buslabeling/%s',gtfiles(i).name));
      % end
      %segs = cellfun2(@(x)x.res.seg,gtmat);
      %seg=sum(cat(3,segs{:}),3);
      seg = gtmat.res.seg;
      gtim = faces2colors(seg);
      
      shower = I;
      fg = repmat(double(seg~=0),[1 1 3]);
      
      shower(find(fg)) = shower(find(fg))*.0 + 1.0*gtim(find(fg));
      gtim = shower;
      gtim(repmat(seg==0,[1 1 3]))=0;
      
    else
      gtim = zeros(size(I,1),size(I,2));
    end
    
    if 0 %size(exemplar_overlay.friendbb,1) == 0
      counter = counter + 1;
      continue
    end
    allbb(zzz,:)
    figure(1)
    clf
    NR=show_hits_figure_iccv(models,allbb(zzz,:),I, ...
                          exemplar_overlay.segI,exemplar_overlay,gtim);

    drawnow
    


%     output.I = I;
%     output.result = exemplar_overlay.segI;
%     output.WARNINGgt = gtim;
    
%     try
%       dfile = sprintf('/nfs/onega_no_backups/users/ashrivas/buses/%s.mat',curid);
%       dresult = load(dfile);
%       output.dresult = dresult.rim;
%     catch
%       output.dresult = []; %2*ones(size(output.I,1),size(output.I,2));
%     end
    
%     save(sprintf('/nfs/baikal/tmalisie/iccvres/finalbusmats/%05d.mat', ...
%                  k),'output');

     if isfield(finalstruct,'basedir')
      filer = sprintf('%s/%s_%05d.pdf',finalstruct.basedir,models{1}.cls,k);
      %      set(gcf,'PaperPosition',[0 0 2 6])
      set(gcf,'PaperPosition',[0 0 2*NR(1) 2*NR(2)],...
        'PaperSize',[2*NR(1) 2*NR(2)]);
      
      print(gcf,'-dpdf',filer);
      filer2 = filer;
      filer2(end-2:end) = 'png';
      print(gcf,'-dpng',filer2);
    else
      allbbs{end+1} = allbb;
      pause
    end
  
    
    %paintfiler = ...
    %     sprintf('%s/%s.%d.%s.png',basedir, ...
    %             curm.curid, curm.objectid, curm.cls);
    % imwrite(Ipaint,paintfiler);

    
    % curm = models{bbox(6)};
    % basedir = '/nfs/baikal/tmalisie/labelme400/www/voc/exemplars/';
    % paintfiler = ...
    %     sprintf('%s/%s.%d.%s.png',basedir, ...
    %             curm.curid, curm.objectid, curm.cls);
    % imwrite(Ipaint,paintfiler);

    %imagesc(I)
    %plot_bbox(bbox,'',color,color);
    
    %plot_bbox(bbs(curb,:));
    %axis image
    %axis off
    %title(sprintf('I=%s\nE=%s.%d s=%.3f os=%.3f',grid{bbs(curb,5)}.curid,models{bbs(curb,6)}.curid,models{bbs(curb,6)}.objectid,bbs(curb,end),moses(curb)));
    counter = counter+1;
    
  end
end
