function wait_until_all_present(files, PAUSE_TIME, invert)
%Wait until all files are present (or all absent), sleep very
%PAUSE_TIME seconds (Defaults to 5)

if length(files) == 0
  return;
end

if ~exist('PAUSE_TIME','var')
  PAUSE_TIME = 5;
end

if ~exist('invert','var')
  TARGET = 0;
elseif invert==1
  TARGET = length(files);
else
  error(sprintf('invert must be absent or 1'));
end

while 1
  missingfile = cellfun(@(x)~fileexists(x),files);
  if sum(missingfile) == TARGET
    break;
  else
    missings = find(missingfile);
    fprintf(1,['%03d File(s) missing [should be %d]', ...
               'waiting %d sec until re-try\n'], ...
               sum(missingfile),TARGET, PAUSE_TIME);
    for q = 1:length(missings)
      fprintf(1,' --missing %s\n',files{missings(q)});
    end
    pause(PAUSE_TIME);
  end
end

