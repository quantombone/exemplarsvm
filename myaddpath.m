function myaddpath(ddd)
dirs = dir([ddd]);

for i = 1:length(dirs)
  if dirs(i).name(1)~='.'
    addpath(genpath([ddd '/' dirs(i).name]));
  end
end
