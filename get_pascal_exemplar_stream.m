function fg = get_pascal_exemplar_stream(VOCopts, set_name, cls, ...
                                                  MAXLENGTH, must_have_seg)
%Create an exemplar stream, such that each element fg{i} contains
%these fields: (I, bbox, cls, curid, [objectid], [anno])
%Make sure the exemplar has a segmentation associated with it if
%must_have_seg is provided

must_have_seg_string = '';
if ~exist('must_have_seg','var')
  must_have_seg = 0;
elseif must_have_seg == 1
  must_have_seg_string = '-seg';
end

%The maximum number of exemplars to process
if ~exist('MAXLENGTH','var')
  MAXLENGTH = 2000;
end

basedir = sprintf('%s/models/streams/',VOCopts.localdir);
if ~exist(basedir,'dir')
  mkdir(basedir);
end
streamname = sprintf('%s/%s-%s-%d-exemplar%s.mat',basedir,set_name,cls,MAXLENGTH,...
                     must_have_seg_string);
if fileexists(streamname)
  fprintf(1,'Loading %s\n',streamname);
  load(streamname);
  return;
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,set_name),...
                  '%s %d');
ids = ids(gt==1);

fg = cell(0,1);

for i = 1:length(ids)
  curid = ids{i};

  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));
  if must_have_seg && (recs.segmented == 0)
    %SKip over unsegmented images
    continue
  end
  filename = sprintf(VOCopts.imgpath,curid);
  
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
    
    %anno is the data-set-specific version
    res.anno = recs.objects(objectid);
    
    res.filer = sprintf('%s.%d.%s.mat', curid, objectid, cls);

    fg{end+1} = res;
    
    if length(fg) == MAXLENGTH
      save(streamname,'fg');
      return;
    end
  end
end

save(streamname,'fg');