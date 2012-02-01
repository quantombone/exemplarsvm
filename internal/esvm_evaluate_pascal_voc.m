function results = esvm_evaluate_pascal_voc(test_struct, test_set, ...
                                            models, params, test_set_name)

% Evaluate PASCAL VOC detection task with the models, their output
% firings grid, on the set target_directory which can be either
% 'trainval' or 'test'
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

params.set_name = test_set_name;

models_name = models{1}.models_name;
cls = models{1}.cls;

%VOCopts = params.dataset_params;

%In case we want to evaluate a subset of detectors
% targets = [4 5];
% for i = 1:length(grid)
%   goods = find(ismember(grid{i}.bboxes(:,6),targets));
%   grid{i}.coarse_boxes = grid{i}.coarse_boxes(goods,:);
%   grid{i}.bboxes = grid{i}.bboxes(goods,:);
%   grid{i}.extras.os = grid{i}.extras.os(goods,:);
% end

if ~exist('models_name','var')
  CACHE_FILE = 0;
  models_name = '';
elseif length(params.localdir)>0
  CACHE_FILE = 1;
else
  CACHE_FILE = 0;
  models_name = '';
end

% has_marker = (target_directory=='+') + ...
%     (target_directory=='-');

% has_marker = find(has_marker);
% if length(has_marker) > 0
%   VOCopts.testset = target_directory(1:has_marker(1)-1);
% else
%   VOCopts.testset = target_directory;
% end

resfile = sprintf('%s/results/%s_%s_results.mat',...
                  params.localdir, models_name, test_set_name);

if CACHE_FILE == 1
  reslock = [resfile '.lock'];
  if fileexists(resfile) || mymkdir_dist(reslock)==0
    %wait until lockfiles are gone
    wait_until_all_present({reslock},5,1);
    fprintf(1,'Pre loading %s\n',resfile);
    res = load_keep_trying(resfile);
    results = res.results;
    return;
  end
end


%% Avoid writing a file to disk
%mname = sprintf('%s%s',models_name,test_struct.calib_string);
%filer = sprintf('%s/results/comp3_det_%s.txt',...
%                params.localdir,models_name);

%Create directory if it is not present
%[aaa,bbb,ccc] = fileparts(filer);
%if CACHE_FILE && ~exist(aaa,'dir')
%  mkdir(aaa);
%end

% fprintf(1,'Writing File %s\n',filer);
% filerlock = [filer '.lock'];
% if fileexists(filer) || (mymkdir_dist(filerlock)==0)
%   wait_until_all_present({filerlock},5,1);
% else
%   fid = fopen(filer,'w');
%   for i = 1:length(test_struct.final_boxes)
%     for q = 1:size(test_struct.final_boxes{i},1)
%       fprintf(fid,'%s %f %f %f %f %f\n',curid,...
%               test_struct.final_boxes{i}(q,end),...
%               test_struct.final_boxes{i}(q,1:4));
%     end
%   end
%   fclose(fid);
% end

%make sure filer is present in order to continue here
%if CACHE_FILE
%  wait_until_all_present({filer});
%end

%fprintf(1,'HACK: changing OVERLAP HERE!\n');
%VOCopts.minoverlap = .4;

BB = cat(1,test_struct.final_boxes{:});

% goods = find(cellfun(@(x)size(x,1),test_struct.final_boxes));

% BB = cellfun2(@(x)x(:,1:4),test_struct.final_boxes(goods));
% BB = cat(1,BB{:});
% BB = BB';

% conf = cellfun2(@(x)x(:,end),test_struct.final_boxes(goods));
% conf = cat(1,conf{:});

% superids = cellfun2(@(x)x(:,11),test_struct.final_boxes(goods));
% superids = cat(1,superids{:});
% ids = cell(length(test_struct.final_boxes(goods)),1);

%for i = 1:length(test_struct.final_boxes(goods))
%  ids{i} = repmat({grid{i}.curid},size(test_struct.final_boxes{goods(i)},1),1);
%end
%ids = cat(1,ids{:});

figure(2)
clf

%draw = 1;
%NOTE(TJM): this is really close to working without writing a local
%evaluation file, and needs to be fixed soon
[results.recall,results.prec,results.ap,results.apold,results.fp,results.tp,results.npos,results.corr] = VOCevaldet(test_set,BB,cls,params);

%if CACHE_FILE && exist(filerlock,'dir')
% rmdir(filerlock);
%end

set(gca,'FontSize',16)
set(get(gca,'Title'),'FontSize',16)
set(get(gca,'YLabel'),'FontSize',16)
set(get(gca,'XLabel'),'FontSize',16)
axis([0 1 0 1]);
results.cls = cls;
drawnow

filer = sprintf(['%s/results/pr.%s-on-%s.pdf'], ...
                params.localdir, ...
                models_name, ...
                test_set_name);

[basedir,tmp,tmp] = fileparts(filer);

if CACHE_FILE
  if ~exist(basedir,'dir')
    mkdir(basedir);
  end
  
  set(gcf,'PaperPosition',[0 0 8 8])
  print(gcf,'-dpdf',filer);
  filer2 = strrep(filer,'.pdf','.png');
  print(gcf,'-dpng',filer2);
  fprintf(1,'Just Wrote %s\n',filer);

  % results2 = [];
  % results2.ap = results.ap;
  % results2.apold = results.apold;
  
  % save(resfile,'results2');
    
  %TODO: we are saving really large files for exemplarNN
  save(resfile,'results','test_struct');
    
  if exist(reslock,'dir')
    rmdir(reslock);
  end
end
