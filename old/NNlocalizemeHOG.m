function [resstruct,t] = NNlocalizemeHOG(t, sbin, w, b, thresh, TOPK, ...
                                       lpo, SAVE_SVS)
% Localize object in pyramid via sliding windows and (dot product +
% bias recognition score)
% (w,b) are cell matrices which contain learned SVM parameters
% t: input image
% w: cell array of learned templates
% b: cell array of corresponding offsets
% thresh: keep all detections above this threshold
% TOPK: keep at most TOPK detections per exemplars
% lpo: Levels-Per-Octave during search
% resstruct: sliding window output struct
%
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('SAVE_SVS','var')
  SAVE_SVS = 0;
end

fprintf(1,'Localizing %d in I=[%dx%d@%d]',length(w),...
          size(t,1),size(t,2),lpo);
  
% if only one input argument is specified, then just compute the
% pyramid and exit
only_compute_pyramid = 0;
if nargin == 1 && nargout == 1
  only_compute_pyramid = 1;
  w{1} = [];
  b{1} = 0;
end

if ~exist('b','var')
  %if bias is not present, set to zero
  for i = 1:length(w)
    b{i} = 0;
  end
end

if ~exist('thresh','var')
  thresh = -1.05;
end

if ~exist('lpo','var') %levels per octavee
  lpo = 10;
end

if ~isstruct(t)  
  starter=tic;
  I = t;
  clear t
  t.I = I;

  %Compute pyramid
  [t.hog,t.scales] = featpyramid2(t.I, sbin, lpo);
  
  hhh = ceil(1+max(cellfun(@(x)max([size(x,1) size(x,2)]),w))/2);
  
  %fprintf(1,'HACK-no-PAD');
  %hhh = 0;
  
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

%start with smallest level first
normx2 = cellfun(@(x)norm(x(:)).^2,w);

fprintf(1,'\nHACK ordering\n');
for level = 1:length(t.hog) %length(t.hog):-1:1
  featr = t.hog{level};
  
  cellnorms2 = sum(featr.^2,3);
  cellnorms2_ii = cumsum(cumsum(cellnorms2,2),1);

  rootmatch = fconv(featr, w, 1, length(w));

  rmsizes = cellfun2(@(x)size(x), ...
                     rootmatch);

  for exid = 1:length(w)
    if prod(rmsizes{exid}) == 0
      continue
    end

    if size(w{exid},1)*size(w{exid},2) == 1
      curq = cellnorms2;
    else  
      shiftedtop = ...
          (circshift2(cellnorms2_ii,[size(w{exid},1) ...
                    0]));
      
      shiftedleft = ...
          (circshift2(cellnorms2_ii,[0 ...
                    size(w{exid},2)]));
      
      shiftedboth = ...
          (circshift2(cellnorms2_ii,[size(w{exid},1) ...
                    size(w{exid},2)]));
      
      curq = cellnorms2_ii - shiftedtop - shiftedleft + shiftedboth;    
      curq = circshift2(curq,-[size(w{exid},1) size(w{exid},2)]+1);
      
      curq = curq(1:size(rootmatch{exid},1),1:size(rootmatch{exid}, ...
                                                   2));
    end
    
    cur_scores = -(normx2(exid) + curq - 2*rootmatch{exid});
    
    
    %max(cur_scores(:))


    
    hg_size = size(w{exid});

    [aa,indexes] = sort(cur_scores(:),'descend');
    NKEEP = sum((aa>maxers{exid}) & (aa>=thresh));    
    sss = size(w{exid});
    
    [uus,vvs] = ind2sub(rmsizes{exid}(1:2),...
                        indexes(1:NKEEP));
    
    for z = 1:NKEEP
      score_grid{exid}(end+1) = aa(z);
      ip.level = level;
      ip.scale = t.scales(level);
      ip.offset = [uus(z) vvs(z)] - t.padder;
      ip.bb = [([ip.offset(2) ip.offset(1) ip.offset(2)+size(w{exid},2) ...
                 ip.offset(1)+size(w{exid},1)] - 1) * ...
               sbin/ip.scale + 1] + [0 0 -1 -1];
      
      id_grid{exid}{end+1} = ip; 
      
      uu = uus(z);
      vv = vvs(z);
      if SAVE_SVS == 1
        support_grid{exid}{end+1} = ...
            reshape(t.hog{level}(uu+(1:sss(1))-1, ...
                                 vv+(1:sss(2))-1,:), ...
                    prod(hg_size),1);
      end
    end
    
    if (NKEEP > 0)
      newtopk = min(TOPK,length(score_grid{exid}));
      [aa,bb] = psort(-score_grid{exid}',newtopk);
      score_grid{exid} = score_grid{exid}(bb);
      id_grid{exid} = id_grid{exid}(bb);
      if SAVE_SVS == 1
        support_grid{exid} = support_grid{exid}(bb);
      end
      maxers{exid} = min(-aa);
    end   
  end
end

resstruct.score_grid = score_grid;
resstruct.id_grid = id_grid;
if SAVE_SVS == 1
  resstruct.support_grid = support_grid;
else
  resstruct.support_grid = cell(0,1);
end

% resstruct.exgrid = cell(length(w),1);
% for i = 1:length(w)
%   for j = 1:length(id_grid{i})
%     resstruct.exgrid{i}{j} = i;
%   end
% end
