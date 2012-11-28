function [model] = esvm_perform_platt_calibration(data_set, model)
% Perform calibration by learning the sigmoid parameters (linear
% transformation of svm scores) for each model independently. This
% type of SVM classifier calibration is due to John Platt who used
% this trick to convert SVM output scors into probabilities for
% comparison.  If we perform an operation such as NMS, we will now
% have "comparable" scores.  This is performed on the 'trainval' set
% for PASCAL VOC.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

%if enabled, do NMS, if disabled return raw detections
DO_NMS = 0;

boxes = model.boxes;

% if enabled, display images
display = model.params.display;

%model_ids = cellfun2(@(x)x.curid,model.models);
targets = 1:length(model.models);

cls = model.models{1}.cls;
targetc = cls;

OS_THRESH = .5;

for exid = 1:length(model.models)
  fprintf(1,'.');
  
  hits = find((boxes(:,6)==exid));
  b = boxes(hits,:);
  b = esvm_nms(b,.5);
  newb{exid} = b;
  
  %[~,beta] = sort(boxes(hits,end),'descend');
  %hits = hits(beta);
  
  r = VOCevaldet(data_set,b,cls,OS_THRESH);
  newcorrect{exid} = r.is_correct;
  extrascores = ones(100,1)*max(b(:,end));
  extraprec = ones(100,1)*r.prec(1);
  extrascores2 = ones(100,1)*-1;
  extraprec2 = ones(100,1)*0;
  beta = esvm_learn_sigmoid(cat(1,b(:,end),extrascores,extrascores2), ...
                            cat(1, r.prec, ...
                                extraprec,extraprec2));
  betas(exid,:) = beta;
  aps(exid) = r.ap;
  continue


  figure(1)
  clf
  plot(b(:,end),r.prec,'b.-')
  hold all
  plot(b(:,end),1./(1+exp(-beta(1)*(b(:,end)-beta(2)))),'r.')
  title(num2str(r.ap))
  drawnow


  figure(2)
  I = showTopDetections(data_set,b,16);
  imagesc(I);
  title('topdets')
  drawnow
  figure(3)
  I=showTopDetections(model.data_set,model.models{exid}.bb);
  imagesc(I)
  title('exemplar')
  pause
end

boxes = cat(1,newb{:});
boxes = esvm_calibrate_boxes(boxes,betas);
correct = cat(1,newcorrect{:});

model.betas = betas;
model.aps = aps;

model.boxes2 = boxes;
model.correct = correct;
