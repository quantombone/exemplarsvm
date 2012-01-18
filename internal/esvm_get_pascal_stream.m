function fg = esvm_get_pascal_stream(stream_params, dataset_params)
% Create an exemplar stream, such that each element fg{i} contains
% these fields: (I, bbox, cls, curid, filer, [objectid], [anno])
% Make sure the exemplar has a segmentation associated with it if
% must_have_seg is provided
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~exist('dataset_params','var')
  dataset_params.localdir = '';
  CACHE_FILE = 0;
elseif exist('dataset_params','var') && isfield(dataset_params,'localdir') ...
      && length(dataset_params.localdir)>0
  CACHE_FILE = 1;
else
  dataset_params.localdir = '';
  CACHE_FILE = 0;
end

if ~isfield(stream_params,'cls')
  stream_params.cls = '';
end

basedir = sprintf('%s/models/streams/',dataset_params.localdir);
if CACHE_FILE == 1 && ~exist(basedir,'dir')
  mkdir(basedir);
end
streamname = sprintf('%s/%s-%s-%d-%s%s.mat',...
                     basedir,stream_params.stream_set_name,...
                     stream_params.cls,...
                     stream_params.stream_max_ex,...
                     stream_params.model_type,...
                     stream_params.must_have_seg_string);

if CACHE_FILE && fileexists(streamname)
  fprintf(1,'Loading %s\n',streamname);
  load(streamname);
  return;
end


if length(stream_params.cls)>0 && isfield(dataset_params,'clsimgsetpath') ...
      && isfield(stream_params,'stream_set_name')

  %% Load ids of all images in trainval that contain cls
  [ids,gt] = textread(sprintf(dataset_params.clsimgsetpath,stream_params.cls,...
                              stream_params.stream_set_name),...
                      '%s %d');
  ids = ids(gt==1);
  all_recs = cellfun2(@(x)sprintf(dataset_params.annopath,x),ids);
  %BUG(TJM):??? make sure this works on voc training regular style
else
  
  %assume that we don't have cls, we have a cell array of these
  all_recs = cellfun2(@(x)x.recs,stream_params.pos_set);
  all_I = cellfun2(@(x)x.I,stream_params.pos_set);
  ids = cell(size(all_recs));
  for i = 1:length(ids)
    ids{i} = sprintf('%08d',i);
  end
end


fg = cell(0,1);

for i = 1:length(all_recs)
  curid = ids{i};  
  
  if isstr(all_recs{i})
    recs = PASreadrecord(all_recs{i});
  else
    recs = all_recs{i};
  end

  if stream_params.must_have_seg && (recs.segmented == 0)
    %skip over unsegmented images
    continue
  end
    
  if isstr(all_recs{i})
    filename = sprintf(dataset_params.imgpath,curid);
  else
    filename = all_I{i};
  end
  
  if strcmp(stream_params.model_type,'exemplar')
    for objectid = 1:length(recs.objects)
      isinclass = ismember({recs.objects(objectid).class},{stream_params.cls});
      if length(stream_params.cls)==0
        isinclass = 1;
      end
      %skip difficult objects, and objects not of target class
      if (recs.objects(objectid).difficult==1) | ...
            ~isinclass
        continue
      end      
      fprintf(1,'.');
      
      res.I = filename;
      res.bbox = recs.objects(objectid).bbox;
      res.cls = recs.objects(objectid).class;%stream_params.cls;
      res.objectid = objectid;
      res.curid = curid;
      
      %anno is the data-set-specific version
      res.anno = recs.objects(objectid);
      
      res.filer = sprintf('%s.%d.%s.mat', curid, objectid, stream_params.cls);
      
      fg{end+1} = res;
      
      if length(fg) == stream_params.stream_max_ex
        if CACHE_FILE == 1
          save(streamname,'fg');
        end
        return;
      end
    end
  elseif strcmp(stream_params.model_type,'scene')
    fprintf(1,'.');
    
    res.I = filename;
    
    %Use the entire scene (remember VOC stores imgsize in a strange order)
    res.bbox = [1 1 recs.imgsize(1) recs.imgsize(2)];
    
    res.cls = stream_params.cls;
    
    %for scenes use a 1 for objectid
    res.objectid = 1;
    
    %anno is the data-set-specific version
    res.anno = recs.objects;
    
    res.filer = sprintf('%s.%d.%s.mat', curid, res.objectid, stream_params.cls);
    
    fg{end+1} = res;
    
    if length(fg) == stream_params.stream_max_ex
      if CACHE_FILE == 1
        save(streamname,'fg');
      end
      return; 
    end
  else
    error(sprintf('Invalid model_type %s\\n',stream_params.model_type));
  end
end

if CACHE_FILE == 1
  save(streamname,'fg');
end
