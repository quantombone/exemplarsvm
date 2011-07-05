function cleanse_files

d = '/nfs/baikal/tmalisie/finalbuslabeling/';
files = dir([d '*mat']);
NE = 0;
for i = 1:length(files)
  r = load([d files(i).name]);
  if sum(r.res.seg(:)) == 0
    fprintf(1,'error here\n');
    %unix(['rm ' d files(i).name]);
    NE = NE + 1;
  end
end
NE
