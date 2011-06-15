function fg = get_pascal_stream(set, cls)
%Create an exemplar stream, such that each element fg{i} contains
%these fields: (I, bbox, cls, curid, [objectid], [anno])

VOCinit;

basedir = sprintf('%s/%s/',VOCopts.localdir,'streams');
if ~exist(basedir,'dir')
  mkdir(basedir);
end
streamname = sprintf('%s/%s-%s.mat',basedir,set,cls);
if fileexists(streamname)
  fprintf(1,'Loading %s\n',streamname);
  load(streamname);
  return;
end

%% Load ids of all images in trainval that contain cls
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,cls,set),...
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
    
    %anno is the data-set-specific version


    res.I = filename;
    res.bbox = recs.objects(objectid).bbox;
    res.cls = cls;
    res.curid = curid;
    res.objectid = objectid;
    res.anno = recs.objects(objectid);

    fg{end+1} = res;
  end
end

save(streamname,'fg');