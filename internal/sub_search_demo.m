load(['/nfs/baikal/tmalisie/dt-VOC2007-bus/models/bus-g.exemplar-' ...
      'svm/10.dt-bus.mat']);

% Script: PASCAL VOC training/testing script
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

addpath(genpath(pwd));

if ~exist('cls','var')
  cls = 'bus';
end

if ~exist('data_directory','var')
  data_directory = '/Users/tomasz/projects/pascal/';
end

if ~exist('dataset_directory','var')
  dataset_directory = 'VOC2007';
end

if ~exist('results_directory','var')
  results_directory = ...
      sprintf(['/nfs/baikal/tmalisie/dt-%s-%s/'], ...
              dataset_directory, cls);
end

%% Initialize dataset parameters
%data_directory = '/Users/tomasz/projects/Pascal_VOC/';
%results_directory = '/nfs/baikal/tmalisie/esvm-data/';

%data_directory = '/csail/vision-videolabelme/people/tomasz/VOCdevkit/';
%results_directory = sprintf('/csail/vision-videolabelme/people/tomasz/esvm-%s/',cls);

dataset_params = esvm_get_voc_dataset(dataset_directory, ...
                                      data_directory, ...
                                      results_directory);
dataset_params.display = 1;
%dataset_params.dump_images = 1;

params = m.mining_params;
params.detect_keep_threshold = -2;

cur_set = esvm_get_pascal_set(dataset_params, ['trainval+bus']);
for i = 80:length(cur_set)
  Ibase = toI(cur_set{i});
  sbase = size(Ibase);
  bests = -100;
  
  for j = 1:100

    s = round(sbase+(.5*rand(1,3).*s));
    s = max(50,round(s + randn(size(s))*200));
    s = min(s,500);
    I = max(0.0,min(1.0,imresize(Ibase,[s(1) s(2)])));
    
    tic
    rs = esvm_detect(I,{m},params);
    toc
    
    curs = rs.bbs{1}(1,end);
    if curs > bests
      sbase = s;
      bests = curs;
      
      figure(1)
      clf
      imagesc(I)
      if size(rs.bbs{1},1)>0
        plot_bbox(rs.bbs{1}(1,:),num2str(rs.bbs{1}(1,end)));
        title(num2str(rs.bbs{1}(1,end)))
      end
      drawnow
      
    end
  end
end

