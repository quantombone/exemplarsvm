function [w,b,maxscores,maxos] = generate_iccv_graph(g, startid, startlvl)
%Generate a graph given the initial level structure
if ~exist('g','var')
  g = initialize_all_positives;
end

%startid = 9;
%startlvl = 8;
%startlvl = 7;
% startid = 13;
% startlvl = 9;
% startid = 16;
% startlvl = 3;
% startid = 34;
% startlvl = 20;

if ~exist('startid','var')
  startid = 202;
  startlvl = 10;
  startlvl = 1;
  startid = 34;
  startlvl = 1;
  startid = 25;
  startlvl = 10;
end

x = g{startid}.curw{startlvl};
b = 0;

startx = x(:);

if 0 %% GET WIGGLES
  VOCinit;
  I = imread(sprintf(VOCopts.imgpath,g{startid}.curid));
  I = im2double(I);
  w = x-mean(x(:));
  xxx2 = replica_hits(I, g{1}.params.sbin, w, b);
  startx = [xxx2];  
  x = reshape(mean(xxx2,2),size(x));
end

w = x - mean(x(:));

gtmask = g{startid}.curm{startlvl};
masker = gtmask;
w = x / norm(x(:));
%w = x - mean(x(:));
%w = x2w(x);

%initialize with none
m.model.nsv = zeros(prod(size(w)),0);
m.model.svids = [];
m.model.mask = zeros(size(w,1),size(w,2));
%m.model.mask(2:7,2:7)=1;
%m.model.mask(3:end-3,3:end-3)=1;
m.model.mask = m.model.mask*0+1;

%% kill some of the outsides
%masker = zeros(size(x,1),size(x,2));
%masker = masker*0 + 1;
%masker(3,4) = 1;
%masker(3:end-3,3:end-3) = 1;
%w = w.*repmat(masker,[1 1 31]);
%w(1:2,:,:) = 0;
%w(:,1:2,:) = 0;
%w(end-1:end,:,:) = 0;
%w(:,end-1:end,:) = 0;

%w = x2w(x);

cls = g{1}.cls;
bg = get_pascal_bg('train',sprintf('-%s',cls));

mining_queue = initialize_mining_queue(bg);
%myRandomize;
%r = randperm(length(mining_queue));
%mining_queue = mining_queue(r);

%bbs = cellfun2(@(x)x.gt_box,g);
%bbs = cat(1,bbs{:});

%find best overlapping segments
%f = @(x)get_bestos(g{startid}.gt_box,x);
%os = cellfun(@(x)f(x.gt_box),g);
%[sorted_os,order] = sort(os,'descend');

for qqq = 1:10

  xxx = zeros(prod(size(w)),0);
  N = 1;
  if length(m.model.nsv) > 0
    os_thresh = 0;
    %order = order(sorted_os >= os_thresh);
    order = 1:length(g);
    fprintf(1,'Filtered down to %d hypothesis with os>%.3f\n',length(order),os_thresh);

    [maxscores, maxlevels, maxoffsets, maxos, maxfeats] = neighbor_scores(g,w,b,masker);
   
    %N = 1+5*(qqq>1);
    N = 7;
    
    figure(145)
    plot(sort(maxscores,'descend'),'r.');
    title('maxscores')
    drawnow
    rscores = maxscores;% + maxos;
    %rscores(maxos < .7) = -100;
    %rscores(maxscores <= -1) = -100;
    
    sumgood = sum(rscores >- 100);
    N = min(sumgood,N);
    [aa,bb] = sort(rscores,'descend');
    xxx = zeros(prod(size(w)),N);    
    
    figure(44)
    clf
    for i = 1:N
      
      pads = [ceil(size(w,1)/2) ceil(size(w,2)/2) 0];
      curx = padarray(g{order(bb(i))}.curw{maxlevels(bb(i))},pads,0);
      rm = fconv(curx,{w},1,1);
      [value,index] = max(rm{1}(:));
      scores = max(rm{1}(:));
      
      [uu,vv] = ind2sub(size(rm{1}),index);
      xx = curx(uu:uu+size(w,1)-1,vv:vv+size(w,2)-1,:);
      xxx(:,i) = xx(:);
      
      subplot(2,N,i);
      %subplot(ceil(sqrt(N)),ceil(sqrt(N)),i)
      imagesc(HOGpicture(xx))
      subplot(2,N,N+i)
      I=show_g(g,order(bb(i)),maxlevels(bb(i)),maxoffsets(bb(i),:), ...
             size(w));
      imagesc(I)
      axis image
      axis off
      % if 1
      %   box = g{order(bb(i))}.curb{maxlevels(bb(i))};;
      %   H = (box(4)-box(2));
      %   W = (box(3)-box(1));
        
      %   miniW = W / (size(curx,2)-pads(2)*2);
      %   miniH = H / (size(curx,1)-pads(1)*2);
      %   box2([1 2]) = box([1 2]);
      %   box2(1) = box2(1) + (vv-pads(2)-1)*miniW;
      %   box2(2) = box2(2) + (uu-pads(1)-1)*miniH;
      %   box2(3) = box2(1) + miniW*size(w,2);
      %   box2(4) = box2(2) + miniH*size(w,1);
        
      %   I = show_g(g,order(bb(i)),maxlevels(bb(i)));
      %   I = pad_image(I,400);
      %   box2 = round(box2+400);
      %   I = I(box2(2):box2(4),box2(1):box2(3),:);
      %   imagesc(I)
      %   axis image
      %   axis off
      % end
      title(num2str(aa(i)))
    end
    
    drawnow
    pause(.1)
  end


  %% here we mine some negatives
  
  %CVPR11 constant
  %SVMC = .01; %(inside exemplar now)
  
  %maximum #windows per image (per exemplar) to mine
  WPI = 400;
  
  %Levels-per-octave defines how many levels between 2x sizes in pyramid
  lpo = 5;
  
  %Image scale of negative images
  scaler = 1.0;
  
  %Maximum number of negatives to mine before SVM kicks in (this
  %defines one iteration)
  MAXW = 2000;
  thresher = -1.05;
  NITER = 10;
  m.model.w = w;
  m.model.hg_size = size(w);
  m.model.b = b;
  
  
  m.model.x = [startx xxx(:,1:min(size(xxx,2),3))];
  %m.model.x = repmat(startx(:),1,100);
  m.model.params.SVMC = .01;%.01;%.01;
  m.model.params.sbin = g{1}.params.sbin;;
  figure(1)
  maxiter = 1;
  if N==1
    maxiter = 2;
  end
  
  if length(m.model.nsv)>0
    fprintf(1,'pretraining with new x\n');
    [m, tmpqueue] = ...
        mine_negatives(m, {}, MAXW, thresher, ...
                       WPI, lpo, scaler, NITER, m.model.params.SVMC, ...
                       bg);
    fprintf(1,'done pretraining with new x\n');
  end
  

  %maximum number of image-scans during training
  MAX_MINES = 100;
  
  for k = 1:maxiter
    cur_queue = mining_queue(1:MAX_MINES);
    rest_queue = mining_queue(MAX_MINES+1:end);
    [m, cur_queue] = ...
        mine_negatives(m, cur_queue, MAXW, thresher, ...
                       WPI, lpo, scaler, NITER, m.model.params.SVMC, ...
                       bg);


    %svscores = m.model.w(:)'*m.model.nsv - m.model.b;
    %goodsvs = find(svscores >= -1);
    %Isv = get_sv_stack(m.model.svids(goodsvs),bg);
    %figure(343)
    %imagesc(Isv)
    %title('SV stack')
    %drawnow
    mining_queue = cat(2,rest_queue,cur_queue);
  end
  drawnow
  w = m.model.w;
  b = m.model.b;
  
  if 0 %% GET WIGGLES
  VOCinit;
  I = imread(sprintf(VOCopts.imgpath,g{startid}.curid));
  I = im2double(I);
  %bbb = g{startid}.gt_box_padded;
  %I = I(bbb(2):bbb(4),bbb(1):bbb(3),:);
  
  xxx2 = replica_hits(I, m.model.params.sbin, w, b);
  startx = [xxx2];  
  end
end

