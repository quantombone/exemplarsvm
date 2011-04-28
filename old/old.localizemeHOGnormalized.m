function [resstruct,t] = localizemeHOGnormalized(t, w, b, thresh, TOPK, lpo)
% Localize object in pyramid via sliding windows and (dot product +
% offset recognition score)
% (w,b) are cell matrices for which contain learned SVM parameters
% from potentially lots exemplars
% t: input image
% w: cell array of learned templates
% b: cell array of corresponding offsets
% thresh: keep all detections above this threshold
% TOPK: keep at most TOPK detections per exemplars
% lpo: Levels-Per-Octave during search
% resstruct: sliding window output struct
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

fprintf(1,'Localizing %d in I=[%dx%d@%d]',length(w),...
          size(t,1),size(t,2),lpo);
  
only_compute_pyramid = 0;
if nargin == 1 && nargout == 1
  only_compute_pyramid = 1;
  w{1} = [];
  b{1} = 0;
end

if ~exist('b','var')
  b{1} = 0;
end

if ~exist('thresh','var')
  thresh = -1.05;
end

if ~exist('lpo','var') %levels per octate
  lpo = 10;
end

if ~isstruct(t)  
  starter=tic;
  I = t;
  clear t
  t.I = I;

  %Compute pyramid
  [t.hog,t.scales] = featpyramid2(t.I, 8, lpo);

  hhh = ceil(1+max(cellfun(@(x)max([size(x,1) size(x,2)]),w))/2);

  t.padder = hhh; 
  for level = 1:length(t.hog)
    t.hog{level} = padarray(t.hog{level}, [t.padder t.padder 0], 0);
  end
  
  minsizes = cellfun(@(x)min([size(x,1) size(x,2)]),t.hog);

  t.hog = t.hog(minsizes>=hhh);
  t.scales = t.scales(minsizes>=hhh);
  resstruct.scales = t.scales;
  
  t.hog = hog_normalize(t.hog);
  if only_compute_pyramid == 1
    resstruct = t;
    return;
  end
else
  I = t.I;
end

feat2 = t.hog;

%keep top TOPK detection windows per classifier
if ~exist('TOPK','var')
  TOPK = 10;
end

%score grid stores the TOPK scores from each exemplar's firing in
%the image
score_grid = cell(length(w),1);
id_grid = cell(length(w),1);
support_grid = cell(length(w),1);
for q = 1:length(w)
  maxers{q} = -100000;
end
resstruct.padder = t.padder;

for level = length(feat2):-1:1
  featr = feat2{level};

  %fprintf(1,'Processing Level %d/%d:',level,length(feat2));
  %starter = tic;
  
  rootmatch{level} = fconv(featr, w, 1, length(w));
 
  resstruct.rmsizes{level} = cellfun2(@(x)size(x), ...
                                      rootmatch{level});
  
  %% old code only worked when everything was one size
  resstruct.featsizes{level} = size(featr);
  %fprintf(1,' Done in %.3f\n',toc(starter));  
  for exid = 1:length(w)    
    cur_scores = rootmatch{level}{exid} - b{exid};
    
    hg_size = size(w{exid});

    [aa,indexes] = sort(cur_scores(:),'descend');
    NKEEP = sum((aa>maxers{exid}) & (aa>=thresh));    
    sss = size(w{exid});
    
    [uus,vvs] = ind2sub(resstruct.rmsizes{level}{exid}(1:2),...
                        indexes(1:NKEEP));
    
    for z = 1:NKEEP
      score_grid{exid}(end+1) = aa(z);
      ip.level = level;
      ip.index = indexes(z);
      id_grid{exid}{end+1} = ip; 
      
      uu = uus(z);
      vv = vvs(z);
      support_grid{exid}{end+1} = ...
          reshape(feat2{level}(uu+(1:sss(1))-1, ...
                               vv+(1:sss(2))-1,:), ...
                  prod(hg_size),1);
    end
    
    if (NKEEP > 0)
      newtopk = min(TOPK,length(score_grid{exid}));
      [aa,bb] = psort(-score_grid{exid}',newtopk);
      score_grid{exid} = score_grid{exid}(bb);
      id_grid{exid} = id_grid{exid}(bb);
      support_grid{exid} = support_grid{exid}(bb);
      maxers{exid} = min(-aa);
    end   
  end
end

resstruct.score_grid = score_grid;
resstruct.id_grid = id_grid;
resstruct.support_grid = support_grid;
resstruct.exgrid = cell(length(w),1);
for i = 1:length(w)
  for j = 1:length(id_grid{i})
    resstruct.exgrid{i}{j} = i;
  end
end
