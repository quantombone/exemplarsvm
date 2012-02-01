function allbbs = esvm_show_top_dets(test_struct, grid, ...
                                     test_set, models, params, ...
                                     maxk, set_name)
% Show maxk top detections for [models] where [grid] is the set of
% detections from the set [test_set], [test_struct] contains final
% boxes after pooling and calibration. If [params.localdir] is
% present, then results are saved based on naming convention into a
% "results" subfolder. maxk is the number of top detections we show
%
% NOTE: this function requires some cleanup, but is functional
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

allbbs = [];

if length(params.localdir) > 0
  CACHE_FILES = 1;
else
  CACHE_FILES = 0;
end

if ~exist('set_name','var')
  set_name = '';
end

%Default exemplar-inpainting show mode
%SHOW_MODE = 1;

%Default segmentation show mode
%SHOW_MODE = 2;

%Mode to show 
%SHOW_MODE = 3;

suffix = '';

%maxk is the maximum number of top detections we display
if ~exist('maxk','var')
  maxk = 20;
  if strcmp(models{1}.cls,'bus')
    maxk = 100;
  end
end

final_boxes = test_struct.unclipped_boxes;
final_maxos = test_struct.final_maxos;

bbs = cat(1,final_boxes{:});
try
  imids = bbs(:,11);
catch
  imids = [];
end
moses = cat(1,final_maxos{:});

%% sort detections by score
try
  [aa,bb] = sort(bbs(:,end), 'descend');
catch
  aa = [];
  bb = [];
end

%% only good ones now!
%bb = bb(moses(bb)>.5);

if 0
  fprintf(1,'Only segmentation ones here\n');
  %% get ones with segmentation only
  ids = cellfun2(@(x)x.curid,models(bbs(bb,6)));
  %VOCinit
  has_seg = cellfun(@(x)fileexists(sprintf('%s/%s/SegmentationObject/%s.png',...
                                           dataset_params.datadir, ...
                                           dataset_params.dataset,x)),ids);
  bb = bb(find(has_seg));
  test_struct.rc = test_struct.rc(find(has_seg));
  suffix = '.seg';
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

for k = 1:maxk

  try
    corr = test_struct.rc(k);
  catch
    corr = 0;
  end
  if 1 
    if counter > length(bb)
      break;
    end
    
    results_dir = sprintf('%s/results/%s.%s%s/',params.localdir,...
                     set_name, ...
                     models{1}.models_name,test_struct.calib_string);
    if ~exist(results_dir,'dir') && (CACHE_FILES == 1)
      mkdir(results_dir);
    end
    
    filer = sprintf('%s/%05d%s.png',results_dir,k,suffix);
    filerlock = [filer '.lock'];

    if CACHE_FILES && (fileexists(filer) || (mymkdir_dist(filerlock) == 0))
      counter = counter + 1;
      fprintf(1,'Already showed detection # %d, score=%.3f\n', k, bbs(bb(counter),end));
      continue
    end

    fprintf(1,'Showing detection # %d, score=%.3f\n', k, bbs(bb(counter),end));
    allbbs(k,:) = bbs(bb(counter),:);
    
    curb = bb(counter);

    %curid = grid{imids(curb)}.curid;

    I = (convert_to_I(test_set{bbs(bb(counter),11)}));
    
    TARGET_BUS = -1;

    if TARGET_BUS > 0
      gtrecs = PASreadrecord(sprintf(params.annopath,curid));
      businds = find(ismember({gtrecs.objects.class},{'bus'}) & ~[gtrecs.objects.difficult]);
      gtbbs = cat(1,gtrecs.objects.bbox);
      gtbbs = gtbbs(businds,:);
    else
      businds = [];
    end

    bbox = bbs(curb,:);

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
    stuff.rank = counter;
   
    extra = '';
    
    if ismember(models{1}.models_name,{'dalal'})
      extra='-dalal';
    end

    estring = '';
    if exist('betas','var')
      estring = 'withbetas';
    end
    
    %ccc = bbox(6);
    
    %target_image_id = imids(bb(counter));
    %target_cluster_id = bbs(bb(counter),5);
            
    %USE THE RAW DETECTION
    %fprintf(1,' -- Taking Final det score\n');
    allbb = bbs(bb(counter),:);
    
    %CVPR VERSION: use the top local score within a cluster
    %fprintf(1,' -- Finding within-cluster local max\n');
    % allbb = test_struct.raw_boxes{target_image_id};
    % osmat = getosmatrix_bb(allbb,bbs(bb(counter),:));
    % goods = find(osmat>.5);
    % allbb = allbb(goods,:);
    % [alpha,beta] = sort(allbb(:,end),'descend');
    % allbb = allbb(beta,:);
    
    sumI = I*0;
    countI = zeros(size(I,1),size(I,2),1);
    %ooo = cell(0,1);
    
    %mean0 = mean(allbb,1);
    %curoses = getosmatrix_bb(mean0(1,:),allbb);
    
    stuff.I = I;
    stuff.params = params;
    
    clear overlays
    
    for zzz = 1:min(1,size(allbb,1))
      overlays{zzz} = esvm_exemplar_inpaint(allbb(zzz,:), ...
                                            models{allbb(zzz,6)}, ...
                                            stuff);
    end
    
    %sumI = sumI ./ repmat(countI,[1 1 3]);
    
    if TARGET_BUS > -1
      %% load GT for test 
      %gtfiles = dir(sprintf(['/nfs/baikal/tmalisie/buslabeling/' ...
      %                '%s*mat'],curid));
      gtmat = load(sprintf(['/nfs/baikal/tmalisie/finalbuslabeling/' ...
                    '%s.%d.mat'], curid, TARGET_BUS));
      seg = gtmat.res.seg;
      gtim = esvm_faces2colors(seg);
      
      shower = I;
      test_set = repmat(double(seg~=0),[1 1 3]);
      
      shower(find(test_set)) = shower(find(test_set))*.0 + 1.0*gtim(find(test_set));
      gtim = shower;
      gtim(repmat(seg==0,[1 1 3]))=0;
      
    else
      gtim = zeros(size(I,1),size(I,2));
    end
    
    figure(1)
    clf

    current_rank = k;
    NR = esvm_show_transfer_figure(I, models, allbb, ...
                                   overlays, current_rank, corr);
    axis image
    drawnow
    snapnow
    
    if CACHE_FILES == 1
      %print(gcf,'-depsc2',filer);
      print(gcf,'-dpng',filer);
      %finalfile = strrep(filer,'.eps','.pdf');
      %unix(sprintf('zsh "ps2pdf -dEPSCrop -dPDFSETTINGS=/prepress %s %s"',...
      %             filer,finalfile));
      
      %if fileexists(finalfile)
      %  unix(sprintf('rm %s',filer));
      %end
      
      if CACHE_FILES && exist(filerlock,'dir')
        rmdir(filerlock);
      end
    end
    
    %filer2 = filer;
    %filer2(end-2:end) = 'png';
    %print(gcf,'-dpng',filer2);
       
    counter = counter+1;    
  end
end
