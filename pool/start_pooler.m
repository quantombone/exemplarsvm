function start_pooler(setname,NPROC)
%Spawn a worker which handles some images in RAM and waits for jobs
%in an infinite loop  MUCH FASTER THAN LOCAL IO intensive mining

if ~exist('setname','var')
  setname = 'trainval';
end

if ~exist('NPROC','var')
  NPROC = 200;
end
VOCinit;

%A directory where the workers register themselves
%BASEDIR = '/nfs/baikal/tmalisie/pool/';

%Get the directory used for worker pools
BASEDIR = get_pool_directory;

%do learning with data from the trainset
fg = get_pascal_bg(setname);

IMS_PER_CHUNK = floor(length(fg)/NPROC);
iii = 1:length(fg);

cuts = round(linspace(1,length(fg)+1,NPROC+1));
inds = cell(NPROC,1);
for i = 1:length(cuts)-1
  inds{i} = iii(cuts(i):cuts(i+1)-1);
end

lens = cellfun(@(x)length(x),inds);
fprintf(1,'NPROC = %d\n',NPROC);
fprintf(1,'length(inds)= %d\n',length(inds));
fprintf(1,'target IMS_PER_CHUNK = %d\n',IMS_PER_CHUNK);
fprintf(1,'real IMS_PER_CHUNK = %.3f, min=%.3f, max=%.3f\n',mean(lens),min(lens),max(lens));

targets = 1:length(inds);
myRandomize;
rrr = randperm(length(targets));
targets = targets(rrr);

localizeparams.thresh = -1;
localizeparams.TOPK = 10;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;
localizeparams.FLIP_LR = 1;
localizeparams.NMS_MINES_OS = 0.5;

nomodels{1}.model.w = zeros(1,1,features);
nomodels{1}.model.w = randn([1 1 features]);
nomodels{1}.model.b = 0;
nomodels{1}.model.params.sbin = 8;

[tmp, me] = unix('hostname');
N = length(targets);

%A data chunks, tells us the following shared info

%Number of processes in pool, to makes files like 0001-00100
data.N = N;
data.BASEDIR = BASEDIR;

%the parameters are shared
data.localizeparams = localizeparams;


for i = 1:N
  current_target = targets(i);
  filer = sprintf('%s/%05d.process',BASEDIR,targets(i));
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
    [tmp,t] = localizemeHOG(I, nomodels, localizeparams);
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

  %the subset of the datset bg for the current set of inds
  data.current_target = current_target;
  %the actual pyramids
  data.ts = ts;
  
  %the gt-recs from loading Ground Truth bounding box files
  data.recs = recs;

  data.subg = subg;
  data.inds = curi;

 
  while 1
      status = ...
          wait_for_chunk(data);
    
    if status == 0
      fprintf(1,'Nothing to do, sleeping\n');
      pause(5);
      
      %write the process file to let world know we are still breathing
      fid = fopen(filer,'w');
      fprintf(fid,me);
      fclose(fid);

    end
  end  
end

function status = wait_for_chunk(data)
%Given a set of precomputed hog pyramids, wait for a classifier
%file, then fire it on everything to produce results

prefix = sprintf('%05d-%05d',data.current_target,data.N);

% filer = '/nfs/baikal/tmalisie/default_poolstart.txt';
% if fileexists(filer)
%   fid = fopen(filer,'r');
%   QUEUEDIR = fscanf(fid,'%s');
%   fclose(fid);
%   fprintf(1,'Loading default queue dir from file %s\n',filer);    
% else
%   fprintf(1,'No default file %s, using hardcoded QUEUEDIR\n',filer);
%   QUEUEDIR = '/nfs/baikal/tmalisie/pool/';
%   cls = 'train';
% end
fprintf(1,'queuedir is %s\n',data.BASEDIR);

DONEDIR = [data.BASEDIR '/pool/'];
fprintf(1,'donedir is %s\n',DONEDIR);

if ~exist(DONEDIR,'dir')
  mkdir(DONEDIR);
end
files = dir([data.BASEDIR '*mat']);

status = 0;
for i = 1:length(files)
  donefile = [DONEDIR prefix '.' files(i).name];
  startfile = [data.BASEDIR files(i).name];
  
  if fileexists(donefile)
    continue
  else
    %%if we are here, then we have a startfile but no endfile
    status = process_file(startfile,donefile,data);
    return;
  end
end

function status = process_file(startfile,donefile,data)
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
mining_params.thresh = data.localizeparams.thresh;
mining_params.TOPK = data.localizeparams.TOPK;
mining_params.lpo = data.localizeparams.lpo;
mining_params.SAVE_SVS = 1;
mining_params.FLIP_LR = data.localizeparams.FLIP_LR;

%%NOTE: this should be carefully set to not blow things up too heavily
mining_params.MAX_WINDOWS_BEFORE_SVM = 2000;


mining_queue = ...
    initialize_mining_queue(data.ts,1:length(data.ts));

chunkstart = tic;
[hn,mining_queue] = load_hn_fg(models, mining_queue, ...
                  data.ts, mining_params);
%Save the time it took for the mining operation
timing = toc(chunkstart);

%% fill in the overlaps from the data
VOCinit;

MAXDETS = 100;

for i = 1:length(hn.objids)
  for j = 1:length(hn.objids{i})
    
    %% take values from data.inds

    recid = hn.objids{i}{j}.curid;
    r = cat(1, ...
            data.recs{recid}.objects.bbox);
    os = ...
        getosmatrix_bb(hn.objids{i}{j}.bb,r);
    [maxos,maxind] = max(os);
    maxclass = ...
        find(ismember(VOCopts.classes,...
                      {data.recs{recid}.objects(maxind).class}));
    hn.objids{i}{j}.maxos = maxos;
    hn.objids{i}{j}.maxclass = maxclass;
    hn.objids{i}{j}.maxind = maxind;
    hn.objids{i}{j}.maxbb = r(maxind,:);
    hn.objids{i}{j}.curid = data.inds(hn.objids{i}{j}.curid);
  end
  
  scores = models{i}.model.w(:)'*hn.xs{i} - models{i}.model.b;
  [tmp,bb] = sort(scores,'descend');
  hits = bb(1:min(length(bb),MAXDETS));
  hn.xs{i} = hn.xs{i}(:,hits);
  hn.objids{i} = hn.objids{i}(hits);
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

%keep only the top 100 dets from this chunk


save(donefile,'hn','timing','mining_queue');
status = 1;
rmdir(lockfile);
