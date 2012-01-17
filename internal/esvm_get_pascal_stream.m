function fg = esvm_get_pascal_stream(stream_params, dataset_params)
%Create an exemplar stream, such that each element fg{i} contains
%these fields: (I, bbox, cls, curid, filer, [objectid], [anno])
%Make sure the exemplar has a segmentation associated with it if
%must_have_seg is provided

if ~isfield(stream_params,'cache_file')
  stream_params.cache_file = 0
end

basedir = sprintf('%s/models/streams/',dataset_params.localdir);
if stream_params.cache_file == 1 && ~exist(basedir,'dir')
  mkdir(basedir);
end
streamname = sprintf('%s/%s-%s-%d-%s%s.mat',...
                     basedir,stream_params.stream_set_name,...
                     stream_params.cls,...
                     stream_params.stream_max_ex,...
                     stream_params.model_type,...
                     stream_params.must_have_seg_string);

if stream_params.cache_file && fileexists(streamname)
  fprintf(1,'Loading %s\n',streamname);
  load(streamname);
  return;
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(dataset_params.clsimgsetpath,stream_params.cls,...
                            stream_params.stream_set_name),...
                    '%s %d');
ids = ids(gt==1);

fg = cell(0,1);

for i = 1:length(ids)
  curid = ids{i};

  recs = PASreadrecord(sprintf(dataset_params.annopath,curid));
  if stream_params.must_have_seg && (recs.segmented == 0)
    %SKip over unsegmented images
    continue
  end
  filename = sprintf(dataset_params.imgpath,curid);
  
  if strcmp(stream_params.model_type,'exemplar')
    for objectid = 1:length(recs.objects)
      %skip difficult objects, and objects not of target class
      if (recs.objects(objectid).difficult==1) | ...
            ~ismember({recs.objects(objectid).class},{stream_params.cls})
        continue
      end
      fprintf(1,'.');
      
      res.I = filename;
      res.bbox = recs.objects(objectid).bbox;
      res.cls = stream_params.cls;
      res.objectid = objectid;
      res.curid = curid;
      
      %anno is the data-set-specific version
      res.anno = recs.objects(objectid);
      
      res.filer = sprintf('%s.%d.%s.mat', curid, objectid, stream_params.cls);
      
      fg{end+1} = res;
      
      if length(fg) == stream_params.stream_max_ex
        if stream_params.cache_file == 1
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
      if stream_params.cache_file == 1
        save(streamname,'fg');
      end
      return; 
    end
  else
    error(sprintf('Invalid model_type %s\\n',stream_params.model_type));
  end
end

if stream_params.cache_file == 1
  save(streamname,'fg');
end
