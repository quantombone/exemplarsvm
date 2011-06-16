function fg = get_pascal_exemplar_stream(set_name, cls, VOCopts, MAXLENGTH)
%Create an exemplar stream, such that each element fg{i} contains
%these fields: (I, bbox, cls, curid, [objectid], [anno])

basedir = sprintf('%s/models/exemplar-streams/',VOCopts.localdir);
if ~exist(basedir,'dir')
  mkdir(basedir);
end
streamname = sprintf('%s/%s-%s.mat',basedir,set_name,cls);
if fileexists(streamname)
  fprintf(1,'Loading %s\n',streamname);
  load(streamname);
  return;
end

%The maximum number of exemplars to process
if ~exist('MAXLENGTH','var')
  MAXLENGTH = 1000000;
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,set_name),...
                  '%s %d');
ids = ids(gt==1);

fg = cell(0,1);

for i = 1:length(ids)
  curid = ids{i};

  recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
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