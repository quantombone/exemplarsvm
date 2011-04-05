function [final,ids,y,catnames] = sun_initialize(setname)
%% Scring to initialize one of the folds fron the SUN397 dataset
%% Tomasz Malisiewicz (tomasz@cmu.edu)

EX_PER_CHUNK = 100;

%This is where Santosh put the SUN297 dataset
sundir = '/nfs/hn38/users/sdivvala/Datasets/SUN397/';

if ~exist('setname','var')
  setname = 'Training_01';
  %setname = 'Testing_01';
end

ids = textread(sprintf('/nfs/hn38/users/sdivvala/Datasets/SUN397/Partitions/%s.txt',setname),'%s');

resdir = sprintf('/nfs/baikal/tmalisie/sun/sunres-%s/', ...
                 setname);
if ~exist(resdir,'dir')
  mkdir(resdir);
end

resfile = [resdir(1:end-1) '.mat'];
if nargout > 0
  if fileexists(resfile)
    fprintf(1,'Loading results from %s\n',resfile);
    load(resfile);
    return;
  end
end
  

for i = 1:length(ids)
  ids{i} = [sundir ids{i}(2:end)];
end


%% Chunk the data into EX_PER_CHUNK exemplars per chunk so that we
%process several images, then write results for entire chunk
inds = do_partition(1:length(ids),EX_PER_CHUNK);
if nargout == 0
  myRandomize;
  r = randperm(length(inds));
else
  r = 1:length(inds);
end


%%TODO: hardcoded feature size here
final = zeros(1984,length(ids));
c = 0;
for iii = 1:length(inds)
  i = r(iii);
  filer = sprintf('%s/%08d.mat', ...
                    resdir,i);
  
  if nargout == 0
    filerlock = [filer '.lock'];
    if fileexists(filer) || (mymkdir_dist(filerlock)==0)
      continue
    end
  end

  fprintf(1,'.');
  if nargout > 0
    res = load(filer);
    final(:,(1:size(res.ms,2))+c) = res.ms;
    c = c + size(res.ms,2);
    continue;
  end
  
  ms = get_all_features(ids(inds{i}));
  msinds = inds{i};
  save(filer,'ms','msinds');
  if exist(filerlock,'dir')
    rmdir(filerlock);
  end
end

[y,catnames] = get_sun_categories(ids);

if nargout > 0
  fprintf(1,'Saving results to %s\n',resfile);
  save(resfile,'final','ids','y','catnames');
end




function ms = get_all_features(fg)
%Compute the features for all of these images

for i=1:length(fg)
  fprintf(1,'.');
  I = convert_to_I(fg{i});
  
  %compute the descriptor by resizing image to 200x200 and applying
  %pedro's features with a sbin=20 to give a 8x8x31 descriptor
  I = imresize(I,[200 200]);
  I = max(0.0,min(1.0,I));
  ms{i} = features(I,20);
end
ms = cellfun2(@(x)reshape(x,[],1),ms);
ms = cat(2,ms{:});

