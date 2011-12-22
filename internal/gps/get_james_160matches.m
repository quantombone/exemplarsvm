function bg = get_james_160matches(index)
%Create a virtual dataset from one of james hays' 78 scene
%completions

%where james stored his mat files
BASEDIR = ['/nfs/baikal/jhhays/scene_completion/' ...
           'old_inpainting_matches/'];

files = dir([BASEDIR '/*mat']);

res = load([BASEDIR files(index).name]);

for i = 1:length(res.best_indices)
  bg{i} = sprintf('load_james_image(%d)',res.best_indices(i));
end
