function search_within_image(baseI,models)

localizeparams.thresh = -1;
localizeparams.TOPK = 1;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 0;
    
for q = 1:100
  I = baseI;
  R = [1 0 0; 0 1 0; 0 0 1];

  %R(1:2,1:2) = eye(2,2)+randn*3;
  %[u,w,v] = svd(R(1:2,1:2));
  %R(1:2,1:2) = u;
  R(1,2) = -.005*q;
  % R(2,1)
  tform = maketform('affine',R);
  I = imtransform(I,tform);
  %I = circshift2(I,round([-5+rand*10 -5+rand*10]));
  
  starter=tic;
  [rs,t] = localizemeHOG(I,models,localizeparams);
  
  scores = cat(2,rs.score_grid{:});
  [aa,bb] = max(scores);
  fprintf(1,' took %.3fsec, maxhit=%.3f, #hits=%d\n',...
          toc(starter),aa,length(scores));
  
  %extract detection box vectors from the localization results
  [coarse_boxes] = extract_bbs_from_rs(rs, models);
  
  boxes = coarse_boxes;
  %map GT boxes from training images onto test image
  boxes = adjust_boxes(coarse_boxes,models);
  
  if size(boxes,1) == 0
    continue
  end
  [aa,bb] = max(boxes(:,end));
  figure(1)
  clf
  I = max(0.0,min(1.0,I));
  imagesc(I)
  plot_bbox(boxes(bb,:))
  axis image
  title(boxes(bb,end))
  drawnow
  
  % boxes = boxes(bb,:);
  % boxes(1:4) = round(boxes(1:4));

  % subI = I(boxes(2):boxes(4),boxes(1):boxes(3),:);
  % Iex = get_exemplar_icon({m},1);
  % if boxes(end-1)==1
  %   Iex = flip_image(Iex);
  % end
  % keyboard
end