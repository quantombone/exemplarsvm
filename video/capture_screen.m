function [Is,xs,scores,bbs] = capture_screen(NITER,models,TOPK,M)
% Capture NITER frames from the screen (run initialize_screenshot first!)
% Show detections from models and keep top TOPK images with those
% detections
% --inputs--
% NITER:      number of frames to capture
% [models]:   the models to fire in the screenshot
% [TOPK]:     the number of topk images to keep (default is NITER)
% --outputs--
% Is:         cell array of TOPK images
% xs:         best detections from models{1}
% scores:     the output detection scores
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('TOPK','var')
    TOPK = NITER;
end

if 0 %%exist('models','var')
    %re-normalize weights
  for i = 1:length(models)
    poshits = mean(models{i}.model.w(:)'*models{i}.model.x,2);
    neghits = mean(models{i}.model.w(:)'*models{i}.model.nsv, ...
                   2);
    
    [alphavec] = pinv([poshits -1; neghits -1])*[1 -1]';
    models{i}.model.w = models{i}.model.w*alphavec(1);
    models{i}.model.b = alphavec(2);
  end
end

xs = [];
scores = [];

if exist('models','var')
    ws = cellfun2(@(x)x.model.w,models);
    bs = cellfun2(@(x)x.model.b,models);
end

%load the screen location and size (from where screenshot is taken)
load subber.mat

initialize_screenshot;

       
xs = cell(0,1);
scores = zeros(0,1);
Is = cell(0,1);
bbs = cell(0,1);

if ~exist('NITER','var')
    NITER = 20;
end

figure(1)

%stuff = get_pascal_bg('test','bus');

for i = 1:NITER
    

    %keep looping/pausing until a new screenshot is obtained!
    %NOTE: this will hang for a long time if the screen doesn't change
    for aaa = 1:100
        %I = convert_to_I(stuff{i});
      I = ScreenCapture(subber);  
      I = im2double(I);
      sizer = size(I);
      ms = max(sizer(1:2));
      
      
      %I = imresize(I,sizer(1:2)*300/ms);
      %I = max(0.0,min(1.0,I));
      
      if length(Is)>=1
        normer = norm(Is{end}(:)-I(:));
        if normer > 0
          break;
        end
      end

      %keep first one
      if length(Is) == 0
        break;
      end
      
      pause(.3)
    end

     
  Is{end+1} = I;
  
  if exist('models','var')
    
    localizeparams.thresh = -1.0;
    localizeparams.TOPK = 10;
    localizeparams.lpo = 20;
    localizeparams.SAVE_SVS = 1;
    
    
    [rs,t] = localizemeHOG(I,models,localizeparams);
    
    [coarse_boxes,scoremasks] = extract_bbs_from_rs(rs,models);
    
    %map GT boxes from training images onto test image
    bb = adjust_boxes(coarse_boxes,models);
    
    bb = nms_within_exemplars(bb,.5);
    
    xraw = get_box_features(bb, size(M.C,1), M.neighbor_thresh);

    res2 = apply_boost_M(xraw,bb,M);

    if length(res2)>0
      res2
    end
    
    bb(:,end) = res2;
    
    bbs{end+1} = bb;
    %bb = nms(bb,.5);
    if sum(size(rs.support_grid{1}))>0
      xs{end+1}= rs.support_grid{1}{1};
      scores(end+1) = rs.score_grid{1}(1);
    else
      xs{end+1} = models{1}.model.w(:)*0;
      scores(end+1) = -2.0;
    end
    
  end
  
  %just keep the TOPK images
  if length(xs) > TOPK
      [alpha,beta] = sort(scores,'descend');
      goods = beta(1:min(length(beta),TOPK));
      Is = Is(goods);
      xs = xs(goods);
      scores = scores(goods);
      bbs = bbs(goods);
  end

  clf
  subplot(2,1,1)
  imagesc(I);
  titler = num2str(i);

  axis image
  axis off
  if exist('models','var') && size(bb,1)>0
      titler = [titler  ' ' num2str(bb(1,end))];
      
  else
      subplot(2,1,2)
      imagesc(I)
      axis image
      axis off
      h=title(titler);
      set(h,'FontSize',20);
      drawnow
      continue;  
  end
    
  bbsave = bb;
  title(titler);
  if size(bb,1)>0
      
      bb = nms(bb,.5);
      sc = max(-1.0,min(1.0,(bb(:,end))));
      g = 1+floor(((sc+1)/2)*20);
      colors = jet(21);
      for i = 1:size(bb,1)
          col1 = colors(g(i),:);
          plot_bbox(bb(i,:),'',col1,col1);
      end
  
      PADDER=400;
      I2 = pad_image(I,PADDER);
      bb(:,1:4) = round(bb(:,1:4)+PADDER);
      
      subplot(2,1,2)
  
      cs = get_exemplar_icon(models,bb(1,6));
      
      %bbI = models{bb(1,6)}.gt_box;
      %bbI = models{bb(1,6)}.coarse_box(13,:);
      %cs =
      %csI(cap_range((bbI(2)+1):bbI(4),1,size(csI,1)),cap_range((bbI(1)+1):bbI(3),1,size(csI,2)),:);
  
      aaamask = exp(-((cs(:,:,1)-1).^2+(cs(:,:,2)-1).^2+(cs(:,:,3)- ...
                                                        1).^2));
      
      self = 1-aaamask/max(aaamask(:));
      %self=double(bwmorph((aaamask<.99),'dilate',2));
      
      inserter = imresize(cs,[bb(1,4)-bb(1,2)+1 bb(1,3)-bb(1,1)+1]);
      inserterself = imresize(self,[size(inserter,1) size(inserter, ...
                                                  2)]);
      
      older = I2(bb(1,2):bb(1,4),bb(1,1):bb(1,3),:);
      
      d = sum((inserter-older).^2,3);
      h = exp(-10*d);
          
      
      amask = imresize(h,[size(inserter,1) size(inserter,2)],'nearest');
      amask = amask/max(amask(:));
      amask = repmat(amask,[1 1 3]);
      
      amask = ones(size(amask,1),size(amask,2),3);
      
      %sm = scoremasks{1}.scoremask;

      %sm = sm - min(sm(:));
      %sm = sm / max(sm(:));
      %colors = jet(100);
      %c = colors(round(sm(:)*99+1),:);
      %c = reshape(c,[size(scoremasks{1}.scoremask,1) ...
      %               size(scoremasks{1}.scoremask,2) 3]);
      %inserter = imresize(c,[size(inserter,1) size(inserter,2)],'nearest');

      
      I2(bb(1,2):bb(1,4),bb(1,1):bb(1,3),:) = inserter;%.*amask+(1-amask).*older;
      I2 = pad_image(I2,-PADDER);
      I2 = min(1.0,max(0.0,I2));
      imagesc(I2)
      axis image
      axis off
  end
  drawnow
end
