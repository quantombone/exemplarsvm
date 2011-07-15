function pool_collect

%Collect result files

BASEDIR = get_pool_directory;
finaldir = [BASEDIR '/finalpool/'];
if ~exist(finaldir,'dir')
  mkdir(finaldir);
end

while 1
  allfiles = dir([BASEDIR '/*.mat']);
  fprintf(1,'found %d files \n',length(allfiles));
  
  for giter = 1:length(allfiles)
    
    startfile = allfiles(giter).name;
    finalfile = [finaldir startfile];
    lockfile = [finalfile '.lock'];

    if fileexists(finalfile) || mymkdir_dist(lockfile) == 0
      continue
    end
    fprintf(1,'start of magic\n');
    files = [];
    while 1
      %keep reading files, until we have all of them
      files = dir([BASEDIR '/pool/*' startfile]);
      fprintf(1,'file len is %d\n',length(files));
      if length(files) == 200
        fprintf(1,'found all, breaking\n');
        break;
      else
        fprintf(1,'Found %d files\n',length(files));
        pause(5)
      end
    end
    fprintf(1,'now concat\n');
    %we now have all files
    res = cell(length(files),1);
    for jiter = 1:length(files)
      fprintf(1,'.');
      res{jiter} = load([BASEDIR '/pool/' files(jiter).name]);
    end

    
  end
  
  if fileexists(lockfile)
    rmdir(lockfile)
  end
  
end
