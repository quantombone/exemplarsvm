function [betas,ALL_bboxes] = perform_calibration(models, grid)
% Perform calibration by learning the sigmoid parameters (linear
% transformation of svm scores) for each model independently. If we
% perform an operation such as NMS, we will now have "comparable"
% scores.  This is performed on the 'trainval' set for PASCAL VOC.

% Tomasz Malisiewicz (tomasz@cmu.edu)

%if enabled, do NMS, if disabled return raw detections
DO_NMS = 0;

if DO_NMS == 0
  fprintf(1,'disabled NMS!\n');
end
%only keep detections that have this overlap score with the entire image
OS_THRESH = 0;

if OS_THRESH > 0
  fprintf(1,'WARNING: only keeping detections above OS threshold: %.3f\n',...
          OS_THRESH);
end

if ~exist('models','var')
  models = load_all_exemplars;
end

% if enabled, display images
display = 0;

%hack for trains
%fprintf(1,'hack for trains\n');
%models{1}.cls = 'sheep';

% if display is enabled and dump_images is enabled, then dump images into DUMPDIR
dump_images = 0;

VOCinit;
DUMPDIR = sprintf('%s/%s/%s/',VOCopts.dumpdir,VOCopts.dataset,models{1}.cls);
if ~exist(DUMPDIR,'dir')
  mkdir(DUMPDIR);
end
% show A NxN grid of top detections (if display is turned on)
SHOW_TOP_N_SVS = 10;

if nargin < 1
  fprintf(1,'Not enough arguments, need at least the grid\n');
  return;
end

% final_dir = ...
%     sprintf('%s/betas',VOCopts.localdir);

% if ~exist('fg','var')
%   fprintf(1,'Loading default set of images\n');
%   fg = cat(1,get_pascal_bg('trainval'),get_pascal_bg('test'));
%   setname = 'voc';
  
%   %[bg,setname] = default_testset;
%   %fg = eval(models{1}.fg);
  
%   %fprintf(1,'HACK USING RAND BG\n');
%   %fg = get_james_bg(100000,round(linspace(1,6400000,100000)));
%   %setname = 'sketchfg';
% end

% bg_size = length(eval(models{1}.bg));
setname = 'voc';

if strcmp(setname,'voc')
  target_directory = 'trainval';
  %target_directory = 'train';
  fprintf(1,'Using VOC set so performing calibration with set: %s\n',target_directory);
  
  
  %% prune grid to contain only images from target_directory
  [cur_set, gt] = textread(sprintf(VOCopts.imgsetpath,target_directory),['%s' ...
                    ' %d']);
  
  gridids = cellfun2(@(x)x.curid,grid);
  goods = ismember(gridids,cur_set);
  grid = grid(goods);
  
  % for i = 1:length(grid)
  %   [tmp,curid,tmp] = fileparts(fg{grid{i}.index});
  %   grid{i}.curid = curid;
  % end

  % gridids = cellfun2(@(x)x.curid,grid);

  % goods = ismember(gridids,cur_set);
  % grid = grid(goods);
  
end



% if ~exist(final_dir','dir')
%   mkdir(final_dir);
% end

final_file = ...
    sprintf('%s/betas/%s_betas.mat',...
            VOCopts.localdir,models{1}.cls);

CACHE_BETAS = 0;
if CACHE_BETAS == 1 && fileexists(final_file)
  %fprintf(1,'not loading!!!!!\n')
  %display = 1;
  fprintf(1,'Loading final file %s\n',final_file);
  load(final_file);
  return;
end

for i = 1:length(models)
  if ~isfield(models{i},'curid')
    models{i}.curid = '-1';
  end
end

model_ids = cellfun2(@(x)x.curid,models);

targets = 1:length(models);

cls = models{1}.cls;

VOCinit;
targetc = find(ismember(VOCopts.classes,models{1}.cls));

fprintf(1,'processing boxes\n');
for i = 1:length(grid)    
  if mod(i,100)==0
    fprintf(1,'.');
  end
  cur = grid{i};
  %cur.bboxes = cur.coarse_boxes;
  % Do image-OS pruning, BEFORE NMS
  % this is useful if we are in image detection mode, where we want
  % to retain detections that are close to the entire image
  if OS_THRESH > 0
    curos = getosmatrix_bb(cur.bboxes(:,1:4),cur.imbb);
    cur.bboxes = cur.bboxes(curos>=OS_THRESH,:);
    cur.coarse_boxes = cur.coarse_boxes(curos>=OS_THRESH,:);
  end
  
  if size(cur.bboxes,1) >= 1
    cur.bboxes(:,5) = 1:size(cur.bboxes,1);    
    cur.coarse_boxes(:,5) = 1:size(cur.bboxes,1);    
    if DO_NMS == 1
      cur.bboxes = nms_within_exemplars(cur.bboxes,.5);
      cur.coarse_boxes = cur.coarse_boxes(cur.bboxes(:,5),:);
    end
    if length(cur.extras)>0
      %cur.extras.os = cur.extras.os(cur.bboxes(:,5),:);
      cur.extras.os = cur.extras.maxos(cur.bboxes(:,5));

      cur.extras.os = cur.extras.os.* ...
          reshape((cur.extras.maxclass(cur.bboxes(:,5))==targetc),size(cur.extras.os));

    end
  end
  
  cur.bboxes(:,5) = grid{i}.index;
  cur.coarse_boxes(:,5) = grid{i}.index;
  
  coarse_boxes{i} = cur.coarse_boxes;
  bboxes{i} = cur.bboxes;
   
  %if we have overlaps, collect them
  if length(cur.extras) > 0
    
    %use all objects as ground truth
    %goods = 1:length(cur.extras.cats);
    

    %% find the ground truth examples of the right category
    %goods = find(ismember(cur.extras.cats,cls));
    
    exids = cur.bboxes(:,6);
   
    %if length(goods) == 0
    %  os{i} = zeros(size(bboxes{i},1),1);
    %else
    %  curos = cur.extras.os(:,goods);
    os{i} = cur.extras.maxos; %max(curos,[],2);
    %end    
  else
    os{i} = zeros(size(bboxes{i},1),1);    
  end
  
  scores{i} = cur.bboxes(:,7)';
end
  
ALL_bboxes = cat(1,bboxes{:});
ALL_coarse_boxes = cat(1,coarse_boxes{:});
ALL_os = cat(1,os{:});

if nargout == 2
  betas = [];
  return;
end
%[aa, bb] = sort(ALL_bboxes(:,end), 'descend');

curids = cellfun2(@(x)x.curid,grid);

%bg = cat(1,get_pascal_bg('trainval'),get_pascal_bg('test'));
fprintf(1,'proessing models\n');

for exid = 1:length(models)
  fprintf(1,'.');
  
  sourcegrid = find(ismember(curids,models{exid}.curid));
  if length(sourcegrid) == 0
    sourcegrid = -1;
  end
  %REMOVE SOURCE IMAGE TOO
  %HACK removed
  hits = find((ALL_bboxes(:,6)==exid));%% & (ALL_bboxes(:,5) ~= sourcegrid));
  all_scores = ALL_bboxes(hits,end);
  all_os = ALL_os(hits,:);
  
  good_scores = all_scores(all_os>=.5);
  good_os = all_os(all_os>=.5);
  
  bad_scores = all_scores(all_os<.5);
  bad_os = all_os(all_os<.5);

  %add virtual sample at os=1.0, score=1.0
  %good_os = cat(1,good_os,1.0);
  %good_scores = cat(1,good_scores,1.0);


  if length(good_os) <= 1 || (length(bad_os) ==0)
    beta = [.1 100];
  else

    [aa,bb] = sort(bad_scores,'descend');
    curlen = min(length(bb),10000*length(good_scores));
    try
    bb = bb(round(linspace(1,length(bb),curlen)));
    catch
      keyboard
    end

    bad_scores = bad_scores(bb);
    bad_os = bad_os(bb);
    all_scores = [good_scores; bad_scores];
    all_os = [good_os; bad_os];
    
    [aaa,bbb] = sort(all_scores, 'descend');
    %bbb = bbb(1:min(1000,length(bbb)));
    %all_scores = all_scores(bbb);
    %all_os = all_os(bbb);
    beta = learn_sigmoid(all_scores, all_os);
  end

  %if beta(1)<.001
  %  beta(1) = .001;
  %end  
  betas(exid,:) = beta;

  if (sum(ismember(exid,targets))==0)
    continue
  end
  
  %if exid==221
  %if exid==171
  if 0 %%display == 1
    figure(1)
    clf
    subplot(1,2,1)  
    plot(all_scores,all_os,'r.')
    xs = linspace(min(all_scores),max(all_scores),1000);
    fx = @(x)(1./(1+exp(-beta(1)*(x-beta(2)))));
    
    hold on
    plot(xs,fx(xs),'b','LineWidth',2)
    axis([min(xs) max(xs) 0 1])
    xlabel('SVM score')
    ylabel(sprintf('Max Overlap Score with %s',models{exid}.cls))
    
    title(sprintf('Learned Sigmoid \\beta=[%.3f %.3f]',beta(1), ...
                  beta(2)))
    subplot(1,2,2)
    %subplot(2,2,1)
    if isfield(models{exid},'I')
      Iex = im2double(models{exid}.I);  
    else
      %try pascal VOC image
      Iex = im2double(imread(sprintf(VOCopts.imgpath, ...
                                     models{exid}.curid)));
    end
    imagesc(Iex)
    plot_bbox(models{exid}.gt_box)
    %[aa,bb] = max(all_scores);
    %plot_bbox(ALL_bboxes(hits(bb),1:4),'',[0 1 0]);
    axis image
    axis off

    bbs=ALL_coarse_boxes(hits,:);
    bbs_os = ALL_os(hits,:);
    %[aa,bb] = sort(bbs(:,end)+bbs_os*.2,'descend');
    [aa,bb] = sort(bbs(:,end),'descend');

    bbs_show = bbs(bb,:);
    
    models{exid}.model.svids = {};
    %m = try_reshape(models{exid},bbs_show,100);

    [models{exid}.model.svids,models{exid}.model.nsv] = ...
        extract_svs(bbs_show,100);%,'trainval','');

    figure(445)
    clf
    imagesc(get_sv_stack(models{exid},8))

    
    % figure(446)
    % clf
    % for jjj = 1:25
    %   I = convert_to_I(sprintf(VOCopts.imgpath,...
    %               sprintf('%06d',bbs(bb(jjj),11))));
    %   curbb = bbs_show(jjj,:); %bbs(bb(jjj),:);
    %   subplot(5,5,jjj)
    %   imagesc(I)
    %   plot_bbox(curbb)
    %   axis image
    %   axis off
    % end

    if 0
    VOCinit;
    curdir = sprintf('%s/visual_regression_trainval/',VOCopts.wwwdir);
    if ~exist(curdir,'dir')
      mkdir(curdir);
    end
    filer = sprintf('%s/%05d_trainonly.png',curdir,exid);
    
    drawnow
    set(gcf,'PaperPosition',[0 0 10 10]);
    print(gcf,filer,'-dpng');
    end

    
    pause
  end
  

  if (display == 0)
    continue
  end

  
  %subplot(2,2,1)
  if isfield(models{exid},'I')
    Iex = im2double(models{exid}.I);  
  else
    %try pascal VOC image
    Iex = im2double(imread(sprintf(VOCopts.imgpath, ...
                                   models{exid}.curid)));
  end
  
  
  %bbox = models{exid}.model.coarse_box(13,:);
  bbox = models{exid}.gt_box; %model.coarse_box(13,:);
  Iex = pad_image(Iex,300);
  bbox = bbox+300;
  bbox = round(bbox);
  Iex = Iex(bbox(2):bbox(4),bbox(1):bbox(3),:);
  
  show1 = Iex;
  %imagesc(Iex)

  %axis image
  %axis off
  %title(sprintf('Exemplar %d',exid))

  %subplot(2,2,3)
  

  hogpic = (HOGpicture(models{exid}.model.w));
  
  NC = 200;
  colorsheet = jet(NC);
  dists = hogpic(:);    
  dists = dists - min(dists);
  dists = dists / (max(dists)+eps);
  dists = round(dists*(NC-1)+1);
  colors = colorsheet(dists,:);
  show2 = reshape(colors,[size(hogpic,1) size(hogpic,2) 3]);

  %axis image
  %axis off
  %title('Learned Template')
  %drawnow
    
  all_bb = ALL_bboxes(hits,:);
  all_os = ALL_os(hits);
  
  
  raw_scores = calibrate_boxes(all_bb,betas);
  raw_scores = 1./(1+exp(-raw_scores(:,end)));

  scores = raw_scores + all_os;
  scores(all_os<.5) = -100;
  %scores(raw_scores<.5) = -100;
  

  
  [alpha,beta] = sort(scores,'descend');
  K = sum(alpha>=-1.0);
  K = min(100,K);
  beta = beta(1:K);
  %beta(K+1:end) = beta(K);
  %keyboard

  NNN = max(1,ceil(sqrt(length(beta))));
  %NNN = SHOW_TOP_N_SVS;

  clear III
  clear IIIscores
  III{1} = zeros(100,100,3);
  IIIscores(1) = -10;
  for aaa = 1:NNN*NNN
    III{aaa} = zeros(100,100,3);
    IIIscores(aaa) = -10;
  end
  
  
  for aaa = 1:NNN*NNN
    fprintf(1,'.');
    if aaa > length(beta)
      break
    end
    
    curI = convert_to_I(fg{all_bb(beta(aaa),5)});   
    %curI = imread(sprintf(VOCopts.imgpath,sprintf('%06d', ...
    %                                              all_bb(beta(aaa),5))));
    %curI = im2double(curI);
    
    bbox = all_bb(beta(aaa),:);
 
    curI = pad_image(curI,300);
    bbox = bbox+300;
    bbox = round(bbox);

    %figure(1)
    %imagesc(curI)
    %drawnow
    try
      Iex = curI(bbox(2):bbox(4),bbox(1):bbox(3),:);
    catch
      Iex = rand(100,100,3);
    end
    Iex = max(0.0,min(1.0,Iex));
    III{aaa} = Iex;
    IIIscores(aaa) = all_bb(beta(aaa),end);
  end
  
  sss = cellfun2(@(x)size(x),III(1:max(1,K)));
  meansize = round(mean(cat(1,sss{:}),1));

  III = cellfun2(@(x)min(1.0,max(0.0,imresize(x,meansize(1:2)))), ...
                 III);  
  
  
  IIIstack = cat(4,III{:});
  IIImean = mean(IIIstack,4);
  
  III2 = cell(1,length(III)+3);

  III2(4:end) = III;

  III2{1} = imresize(show1,meansize(1:2));
  III2{2} = imresize(show2,meansize(1:2));
  III2{3} = IIImean;
  III = III2(1:(end-3));
  
  clear Irow
  III = reshape(III,[NNN NNN]);
  for i = 1:NNN
    Irow{i} = cat(1,III{i,:}); 
  end
  
  I = cat(2,Irow{:});
  figure(2)
  clf
  imshow(I)
  [ws1,ws2,ws3] = size(models{exid}.model.w);
  fg_size = length(fg);

  if length(IIIscores)<3
    IIIscores(end+1:3)= -1.1;
  end
  title(sprintf('Wsize=[%d,%d,%d] sbin=%d os=%.3f {|fg|,|bg|}={%d,%d} Top 3 scores (%.3f %.3f %.3f)',ws1,ws2,ws3,...
                models{exid}.model.params.sbin,OS_THRESH,fg_size,bg_size,...
                IIIscores(1),IIIscores(2),IIIscores(3)));
  
  if dump_images == 1
    figure(2)
    filer = sprintf('%s/result.%d.%s.%s.png', DUMPDIR, ...
                    exid,models{exid}.cls,models{exid}.models_name);
    set(gcf,'PaperPosition',[0 0 20 20]);
    print(gcf,filer,'-dpng');
    
  else
    pause
  end  
end

if CACHE_BETAS == 1
  fprintf(1,'Loaded betas, saving to %s\n',final_file);
  save(final_file,'betas');
end
