function [resstruct,t] = localizemeHOG(t, models, localizeparams)
% Localize object in pyramid via sliding windows and (dot product +
% bias recognition score)
% (w,b) are cell matrices which contain learned SVM parameters
% t: input image
% models: a cell array of models
% models{.}.model.w: cell array of learned templates
% models{.}.model.b: cell array of corresponding offsets
% thresh: keep all detections above this threshold
% TOPK: keep at most TOPK detections per exemplars
% lpo: Levels-Per-Octave during search
% resstruct: sliding window output struct
%
% Tomasz Malisiewicz (tomasz@cmu.edu)
if isfield(localizeparams,'FLIP_LR')
  localizeparams = rmfield(localizeparams,'FLIP_LR');
end
[rs1,t1] = localizemeHOGdriver(t,models,localizeparams);
localizeparams.FLIP_LR = 1;
[rs2,t2] = localizemeHOGdriver(t,models,localizeparams);


for q = 1:length(rs1.score_grid)
  rs1.score_grid{q} = cat(2,rs1.score_grid{q},rs2.score_grid{q});
  if numel(rs2.support_grid)>0
    rs1.support_grid{q} = cat(2,rs1.support_grid{q}, ...
                              rs2.support_grid{q});
  end
  rs1.id_grid{q} = cat(2,rs1.id_grid{q},rs2.id_grid{q});
end
% rs1.score_grid = [rs1.score_grid; rs2.score_grid];
% rs1.id_grid = [rs1.id_grid; rs2.id_grid];
% rs1.support_grid = [rs1.support_grid; rs2.support_grid];

resstruct = rs1;
t = [t1  t2];



function [resstruct,t] = localizemeHOGdriver(t, models, localizeparams)
N = length(models);
ws = cellfun2(@(x)x.model.w,models);
bs = cellfun2(@(x)x.model.b,models);

%NOTE: all exemplars in this set must have the same sbin
sbin = models{1}.model.params.sbin;

%if ~exist('SAVE_SVS','var')
%  SAVE_SVS = 0;
%end

fprintf(1,'Localizing %d in I=[%dx%d@%d]',N,...
          size(t,1),size(t,2),localizeparams.lpo);

%if enabled, do NN computation using integral images on cell norms
%instead of just sliding with fconv
%if ~exist('NN_MODE','var')
%  NN_MODE = 0;
%end
  
% if only one input argument is specified, then just compute the
% pyramid and exit
only_compute_pyramid = 0;
if nargin == 1 && nargout == 1
  only_compute_pyramid = 1;
  ws{1} = [];
  bs{1} = 0;
end

if ~isstruct(t)  
  starter=tic;
  
  if isfield(localizeparams,'FLIP_LR') && ...
        (localizeparams.FLIP_LR == 1)
    fprintf(1,'Flip LR\n');
    %flip image lr here...
    I = t;
    for i = 1:3
      I(:,:,i) = fliplr(I(:,:,i));
    end

  else    
    %take unadulterated image
    I = t;
  end
  
  
  clear t
  t.I = I;

  %Compute pyramid
  [t.hog,t.scales] = featpyramid2(t.I, sbin, localizeparams.lpo);
  
  %maxscale = 50/min(size(t.I,1),size(t.I,2));
  %maxscale = .5;
  %t.hog = t.hog(t.scales<=maxscale);
  %t.scales = t.scales(t.scales<=maxscale);

  
  hhh = ceil(1+max(cellfun(@(x)max([size(x,1) size(x,2)]),ws))/2);
  
  %fprintf(1,'HACK-no-PAD');
  hhh = 2;
  t.padder = hhh; 
  for level = 1:length(t.hog)
    t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
  end
  
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]),t.hog);

  t.hog = t.hog(minsizes >= hhh*2);
  t.scales = t.scales(minsizes >= hhh*2);
  resstruct.scales = t.scales;
  
  if only_compute_pyramid == 1
    resstruct = t;
    return;
  end
else
  I = t.I;
end

%score grid stores the TOPK scores from each exemplar's firing in
%the image
score_grid = cell(N,1);
id_grid = cell(N,1);
support_grid = cell(N,1);
for q = 1:N
  maxers{q} = -100000;
end
resstruct.padder = t.padder;

normx2 = cellfun(@(x)norm(x(:)).^2, ws);

% K = ceil(sqrt(length(t.hog)));
% bighog = zeros(size(t.hog{1},1)*K,size(t.hog{1},2)*K,size(t.hog{1}, ...
%                                                   3));

% c = 1;
% for q1 = 1:K
%   for q2 = 1:K

%     bighog((q1-1)*size(t.hog{1},1) + (1:size(t.hog{c},1)),...
%            (q2-1)*size(t.hog{1},2) + (1:size(t.hog{c},2)),:) = ...
%         t.hog{c};

%     c = c + 1;
%     if c >= length(t.hog)
%       break;
%     end
%   end
% end

%t.hog = {bighog};
%start with smallest level first
for level = length(t.hog):-1:1
  featr = t.hog{level};

  %The norm squared integral image is used for euclidean distance computations
  cellnorms2 = sum(featr.^2,3);
  cellnorms2_ii = cumsum(cumsum(cellnorms2,2),1);


  rootmatch = fconvblas(featr, ws, 1, N);
  %rootmatch = fconv(featr, ws, 1, N);

  rmsizes = cellfun2(@(x)size(x), ...
                     rootmatch);

  for exid = 1:N
    if prod(rmsizes{exid}) == 0
      continue
    end

    %% Only do the following stuff if we are in Nearest-Neighbor
    %% Euclidean distance matching mode
    if isfield(models{exid},'NN_MODE') && (models{exid}.NN_MODE ==1)

      if size(ws{exid},1)*size(ws{exid},2) == 1
        curq = cellnorms2;
      else  
        shiftedtop = ...
            (circshift2(cellnorms2_ii,[size(ws{exid},1) ...
                    0]));
        
        shiftedleft = ...
            (circshift2(cellnorms2_ii,[0 ...
                    size(ws{exid},2)]));
        
        shiftedboth = ...
            (circshift2(cellnorms2_ii,[size(ws{exid},1) ...
                    size(ws{exid},2)]));
        
        curq = cellnorms2_ii - shiftedtop - shiftedleft + shiftedboth;    
        curq = circshift2(curq,-[size(ws{exid},1) size(ws{exid},2)]+1);
        
        curq = curq(1:size(rootmatch{exid},1),...
                    1:size(rootmatch{exid},2));
      end
      cur_scores = -(normx2(exid) + curq - 2*rootmatch{exid});
    else
      cur_scores = rootmatch{exid} - bs{exid};
    end

    hg_size = size(ws{exid});

    [aa,indexes] = sort(cur_scores(:),'descend');
    NKEEP = sum((aa>maxers{exid}) & (aa>=localizeparams.thresh));    
    sss = size(ws{exid});
    
    [uus,vvs] = ind2sub(rmsizes{exid}(1:2),...
                        indexes(1:NKEEP));
    
    for z = 1:NKEEP
      score_grid{exid}(end+1) = aa(z);
      ip.level = level;
      ip.scale = t.scales(level);
      ip.offset = [uus(z) vvs(z)] - t.padder;
      ip.bb = [([ip.offset(2) ip.offset(1) ip.offset(2)+size(ws{exid},2) ...
                 ip.offset(1)+size(ws{exid},1)] - 1) * ...
               sbin/ip.scale + 1] + [0 0 -1 -1];
      ip.flip = 0;
      if isfield(localizeparams,'FLIP_LR') && ...
            (localizeparams.FLIP_LR == 1)
        ip.bb = flip_box(ip.bb,size(I));
        ip.flip = 1;
      end
      id_grid{exid}{end+1} = ip; 
      
      uu = uus(z);
      vv = vvs(z);
      if localizeparams.SAVE_SVS == 1
        support_grid{exid}{end+1} = ...
            reshape(t.hog{level}(uu+(1:sss(1))-1, ...
                                 vv+(1:sss(2))-1,:), ...
                    prod(hg_size),1);
      end
    end
    
    if (NKEEP > 0)
      newtopk = min(localizeparams.TOPK,length(score_grid{exid}));
      [aa,bb] = psort(-score_grid{exid}',newtopk);
      score_grid{exid} = score_grid{exid}(bb);
      id_grid{exid} = id_grid{exid}(bb);
      if localizeparams.SAVE_SVS == 1
        support_grid{exid} = support_grid{exid}(bb);
      end
      maxers{exid} = min(-aa);
    end   
  end
end

resstruct.score_grid = score_grid;
resstruct.id_grid = id_grid;
if localizeparams.SAVE_SVS == 1
  resstruct.support_grid = support_grid;
else
  resstruct.support_grid = cell(0,1);
end
  
sizeI = size(I);

%let everybody know we are flipped
if isfield(localizeparams,'FLIP_LR') && ...
      (localizeparams.FLIP_LR == 1)
  for i = 1:length(resstruct.id_grid)
    for j = 1:length(resstruct.id_grid{i})
      resstruct.id_grid{i}{j}.FLIP_LR = 1;
    end
  end
end


