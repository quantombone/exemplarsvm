function start_pooler
%Spawn a worker which handles some images in RAM and waits for jobs
%in an infinite loop  MUCH FASTER THAN LOCAL IO intensive mining

VOCinit;

%do learning with data from the trainset
fg = get_pascal_bg('trainval');
IMS_PER_CHUNK = round(length(fg)/400);
IMS_PER_CHUNK
inds = do_partition(1:length(fg),IMS_PER_CHUNK);
length(inds)
targets = 1:length(inds);
myRandomize;
rrr = randperm(length(targets));
targets = targets(rrr);
BASEDIR = '/nfs/baikal/tmalisie/pool/';

localizeparams.thresh = -1;;
localizeparams.TOPK = 10;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;
localizeparams.FLIP_LR = 1;

nomodels{1}.model.w = zeros(1,1,features);
nomodels{1}.model.w = randn([1 1 features]);
nomodels{1}.model.b = 0;
nomodels{1}.model.params.sbin = 8;

% time1 = [];
% time2 = [];

% I = convert_to_I(fg{1});
% %I = imresize(I,scales(i));
% %tic
% [resstruct,t] = localizemeHOG(I, nomodels, localizeparams);
% time1=[time1 toc];
% time1=[time1 1];
  
% tic
% [resstruct2,t2] = localizemeHOG(t, nomodels, localizeparams);
% time2=[time2 toc];
% figure(1)
% clf
% plot(scales(1:i),time1,'r')
% hold on;
% plot(scales(1:i),time2,'b')
% drawnow
% end

[a,me] = unix('hostname');

LT = length(targets);

for i = 1:length(targets)
  current_target = targets(i);
  filer = sprintf('%s/%05d.process',BASEDIR,targets(i))
  lockfile = [filer '.lock'];
  if fileexists(filer) || mymkdir_dist(lockfile) == 0
    continue
  end

  curi = inds{targets(i)};
  ts = cell(length(curi),1);
  recs = cell(length(curi),1);
  
  tic
  for j = 1:length(curi)
    fname = fg{curi(j)};
    I = convert_to_I(fname);
    [resstruct,t] = localizemeHOG(I, nomodels, localizeparams);
    ts{j} = t;
    [tmp,curid,ext] = ...
        fileparts(fname);
    recs{j} = ...
        PASreadrecord(sprintf(VOCopts.annopath,curid));  
  end
  toc

  subg = fg(curi);
  fprintf(1,'Now we have a pool worker loaded with %d images\n', ...
          length(curi));
  
  %write the process file
  fid = fopen(filer,'w');
  fprintf(fid,me);
  fclose(fid);
  
  rmdir(lockfile);
  
  while 1
      status = ...
          wait_for_chunk(current_target,LT,localizeparams,ts,recs,subg);
    
    if status == 0
      fprintf(1,'Nothing to do, sleeping\n');
      pause(5);
    end
  end
  fprintf(1,['broken loop, why did' ...
             ' I get here?\n']);
  
end

function status = wait_for_chunk(current_target,LT,localizeparams,ts,recs,subg)
%Given a set of precomputed hog pyramids, wait for a classifier
%file, then fire it on everything to produce results

prefix = sprintf('%05d-%05d',current_target,LT);

QUEUEDIR = '/nfs/baikal/tmalisie/pool/';
DONEDIR = ['/nfs/baikal/tmalisie/' ...
           'pool/done/'];
if ~exist(DONEDIR,'dir')
  mkdir(DONEDIR);
end
files = dir([QUEUEDIR '*mat']);

status = 0;
for i = 1:length(files)
  donefile = [DONEDIR prefix '.' files(i).name];
  startfile = [QUEUEDIR files(i).name];
  
  if fileexists(donefile)
    continue
  else
    %%if we are here, then we have a startfile but no endfile
    status = process_file(startfile,donefile,localizeparams,ts,recs,subg);
    return;
  end
end

function status = process_file(startfile,donefile,localizeparams,ts,recs,subg)
%process the detections now

lockfile = [donefile '.lock'];
if fileexists(donefile) || (mymkdir_dist(lockfile) == 0)
  status = 0;  
  return;
end

fprintf(1,'processing file %s\n',startfile);

m = load(startfile);
if isfield(m,'m')
  models = {m.m};
else
  models = m.models;
end

mining_params = get_default_mining_params;
mining_params.thresh = localizeparams.thresh;
mining_params.TOPK = localizeparams.TOPK;
mining_params.lpo = localizeparams.lpo;
mining_params.SAVE_SVS = 1;
mining_params.FLIP_LR = localizeparams.FLIP_LR;

mining_queue = ...
    initialize_mining_queue(ts,1:length(ts));

chunkstart = tic;
[hn] = load_hn_fg(models, mining_queue, ...
                  ts, mining_params);
timing = toc(chunkstart);

keyboard
%% fill in the overlaps from the data
VOCinit;
for i = 1:length(hn.objids)
  for j = 1:length(hn.objids{i})
    recid = hn.objids{i}{j}.curid;
    r = cat(1, ...
            recs{recid}.objects.bbox);
    os = ...
        getosmatrix_bb(hn.objids{i}{j}.bb,r);
    [maxos,maxind] = max(os);
    maxclass = ...
        find(ismember(VOCopts.classes,{recs{recid}.objects(maxind).class}));
    hn.objids{i}{j}.maxos = maxos;
    hn.objids{i}{j}.maxclass = maxclass;
    hn.objids{i}{j}.maxind = maxind;
    
  end
end

%
%for i = 1:length(models)
%  Isv = get_sv_stack(hn.objids{i}, subg, ...
%                                   models{i}, 5, 5);
%  figure(i)
%  clf
%  imagesc(Isv);
%  drawnow
%end

save(donefile,'hn','timing');
status = 1;
rmdir(lockfile);
