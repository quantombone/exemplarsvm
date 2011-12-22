function fg = get_pascal_stream(VOCopts, cls, CACHE_FILE)
%Create an exemplar stream, such that each element fg{i} contains
%these fields: (I, bbox, cls, curid, [objectid], [anno])
%Make sure the exemplar has a segmentation associated with it if
%must_have_seg is provided

if ~exist('CACHE_FILE','var')
  CACHE_FILE = 0;
end

basedir = sprintf('%s/models/streams/',VOCopts.localdir);
if CACHE_FILE == 1 && ~exist(basedir,'dir')
  mkdir(basedir);
end
streamname = sprintf('%s/%s-%s-%d-%s%s.mat',...
                     basedir,VOCopts.stream_set_name,...
                     cls,VOCopts.stream_max_ex,...
                     VOCopts.model_type,...
                     VOCopts.must_have_seg_string);

if CACHE_FILE && fileexists(streamname)
  fprintf(1,'Loading %s\n',streamname);
  load(streamname);
  return;
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,...
                            VOCopts.stream_set_name),...
                    '%s %d');
ids = ids(gt==1);

fg = cell(0,1);

for i = 1:length(ids)
  curid = ids{i};

  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  if VOCopts.must_have_seg && (recs.segmented == 0)
    %SKip over unsegmented images
    continue
  end
  filename = sprintf(VOCopts.imgpath,curid);
  
  if strcmp(VOCopts.model_type,'exemplar')
    for objectid = 1:length(recs.objects)
      %skip difficult objects, and objects not of target class
      if (recs.objects(objectid).difficult==1) | ...
            ~ismember({recs.objects(objectid).class},{cls})
        continue
      end
      fprintf(1,'.');
      
      res.I = filename;
      res.bbox = recs.objects(objectid).bbox;
      res.cls = cls;
      res.objectid = objectid;
      res.curid = curid;
      
      %anno is the data-set-specific version
      res.anno = recs.objects(objectid);
      
      res.filer = sprintf('%s.%d.%s.mat', curid, objectid, cls);
      
      fg{end+1} = res;
      
      if length(fg) == VOCopts.stream_max_ex
        if CACHE_FILE == 1
          save(streamname,'fg');
        end
        return;
      end
    end
  elseif strcmp(VOCopts.model_type,'scene')
    fprintf(1,'.');
    
    res.I = filename;
    
    %Use the entire scene (remember VOC stores imgsize in a strange order)
    res.bbox = [1 1 recs.imgsize(1) recs.imgsize(2)];
    
    res.cls = cls;
    
    %for scenes use a 1 for objectid
    res.objectid = 1;
    
    %anno is the data-set-specific version
    res.anno = recs.objects;
    
    res.filer = sprintf('%s.%d.%s.mat', curid, res.objectid, cls);
    
    fg{end+1} = res;
    
    if length(fg) == VOCopts.stream_max_ex
      if CACHE_FILE == 1
        save(streamname,'fg');
      end
      return; 
    end
  else
    error(sprintf('Invalid model_type %s\\n',VOCopts.model_type));
  end
end

if CACHE_FILE == 1
  save(streamname,'fg');
end
