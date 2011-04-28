function bool = mymkdir_dist(dirName)
%from santosh
[tmp,h] = unix('hostname');
if 0 %strfind(h,'onega')
  fprintf(1,'onega lock skip\n');
  bool = true;
  return;
end
[smesg, smess, smessid] = mkdir(dirName);
bool = ~strcmp(smessid,'MATLAB:MKDIR:DirectoryExists');