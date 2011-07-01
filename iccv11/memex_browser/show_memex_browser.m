function allbbs = show_memex_browser(dataset_params, models, grid, ...
                                     fg, set_name, ...
                                     finalstruct, maxk)
% Show the exemplar-view of the memex browser
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

%maxk is the maximum number of top detections we display
if ~exist('maxk','var')
  maxk = 5;
end

final_boxes = finalstruct.unclipped_boxes;
final_maxos = finalstruct.final_maxos;

bbs = cat(1,final_boxes{:});
imids = bbs(:,11);
exids = bbs(:,6);

moses = cat(1,final_maxos{:});

%% sort detections by score
[aa,bb] = sort(bbs(:,end), 'descend');

%% only good ones now!
%bb = bb(moses(bb)>.5);

wwwdir = sprintf('%s/memex/%s.%s-%s%s/', dataset_params.localdir,...
                 set_name, models{1}.cls, ...
                 models{1}.models_name, finalstruct.calib_string);

if ~exist(wwwdir,'dir')
  mkdir(wwwdir);
end

%% show the exemplar view

filer = sprintf('%s/index.html', wwwdir);
fid = fopen(filer,'w');
fprintf(fid,['<html><head><title>memex browser</title></head><body>\' ...
             'n']);

fprintf(fid,'<table border=1>\n');
for i = 1:length(models)
  fprintf(fid,'<tr>\n');
  [a,curid,ext] = fileparts(models{i}.I);
  imstuff = sprintf(['<img src="http://balaton.graphics.cs.cmu.edu/' ...
                     'sdivvala/.all/Datasets/Pascal_VOC/VOC2007/JPEGImages/%s%s" />'],curid,ext);
  fprintf(fid,'<td>cls=%s models_name=%s\n%s</td>\n',models{i}.cls, ...
          models{i}.models_name,imstuff);
  
  goods = find(exids==i);
  for j = 1:maxk
    b = bbs(goods(j),:);
    
    [a,curid,ext] = fileparts(fg{b(11)});
    imstuff = sprintf(['<img src="http://balaton.graphics.cs.cmu.edu/' ...
                       'sdivvala/.all/Datasets/Pascal_VOC/VOC2007/JPEGImages/%s%s" />'],curid,ext);
    
    fprintf(fid,'<td>imid=%d score=%.3f\n%s</td>\n', b(11), b(end), ...
            imstuff);
  end
  fprintf(fid,'</tr>\n');
  
end

fprintf(fid,'</table>\n');

fprintf(fid,'</body></html>\n');
fclose(fid);

return;
counter = 1;

for k = 1:maxk

  if 1 
    if counter > length(bb)
      break;
    end
    
    
    filer = sprintf('%s/%05d.pdf',wwwdir,k);
    filerlock = [filer '.lock'];
    if fileexists(filer) || (mymkdir_dist(filerlock) == 0)
      counter = counter + 1;
      continue
    end

    fprintf(1,'Top det %d\n', k);
    
    allbbs(k,:) = bbs(bb(counter),:);
    
    curb = bb(counter);
    curid = grid{imids(curb)}.curid;

    I = convert_to_I(fg{grid{imids(curb)}.index});

    businds = [];


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
    
    %ccc = bbox(6);
    
    %target_image_id = imids(bb(counter));
    %target_cluster_id = bbs(bb(counter),5);
            
    %USE THE RAW DETECTION
    fprintf(1,' -- Taking Final det score\n');
    allbb = bbs(bb(counter),:);
    
    %CVPR VERSION: use the top local score within a cluster
    %fprintf(1,' -- Finding within-cluster local max\n');
    % allbb = finalstruct.raw_boxes{target_image_id};
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
    stuff.dataset_params = dataset_params;
    
    clear overlays
    
    for zzz = 1:min(1,size(allbb,1))
      overlays{zzz} = exemplar_inpaint(allbb(zzz,:), ...
                                       models{allbb(zzz,6)}, ...
                                       stuff);
    end
    
    gtim = zeros(size(I,1),size(I,2));
    
    
    figure(1)
    clf

    NR = show_hits_figure_iccv(I,models,allbb, ...
                               overlays);

    drawnow
    set(gcf,'PaperPosition',[0 0 2*NR(1) 2*NR(2)],...
            'PaperSize',[2*NR(1) 2*NR(2)]);
    
    print(gcf,'-dpdf',filer);
    rmdir(filerlock);
    filer2 = filer;
    filer2(end-2:end) = 'png';
    print(gcf,'-dpng',filer2);
       
    counter = counter+1;
    
  end
end
