function [final,ids]=sun_initialize
%% Initialize script which writes out initial model files for all
%% exemplars of a single category from PASCAL VOC trainval set
%% Script is parallelizable
%% Tomasz Malisiewicz (tomasz@cmu.edu)

sundir = '/nfs/hn38/users/sdivvala/Datasets/SUN397/';
[ids] = textread('/nfs/baikal/tmalisie/sunfiles.txt',...
                    '%s');
for i = 1:length(ids)
  ids{i} = [sundir ids{i}(2:end)];
end

EX_PER_CHUNK = 100;
%% Chunk the data into EX_PER_CHUNK exemplars per chunk so that we
%process several images, then write results for entire chunk
inds = do_partition(1:length(ids),EX_PER_CHUNK);
%myRandomize;
%r = randperm(length(inds));
r = 1:length(inds);

final = zeros(1984,length(ids));
c = 0;
for iii = 1:length(inds)
  i = r(iii);
  filer = sprintf('/nfs/baikal/tmalisie/sunres/%08d.mat', ...
                    i);
  
  %filerlock = [filer '.lock'];
  %if fileexists(filer) || (mymkdir_dist(filerlock)==0)
  %  continue
  %end
  
  res = load(filer);
  final(:,(1:size(res.ms,2))+c) = res.ms;
  c = c + size(res.ms,2);
  
  fprintf(1,'.');
  continue;
  
  ms = get_results(ids(inds{i}));
  msinds = inds{i};
  save(filer,'ms','msinds');
  if exist(filerlock,'dir')
    rmdir(filerlock);
  end
end

function ms = get_results(fg)
for i=1:length(fg)
  fprintf(1,'.');
  I = convert_to_I(fg{i});
  I = imresize(I,[200 200]);
  I = max(0.0,min(1.0,I));
  ms{i} = features(I,20);
end
ms = cellfun2(@(x)reshape(x,[],1),ms);
ms = cat(2,ms{:});
