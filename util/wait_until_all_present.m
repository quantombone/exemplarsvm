function wait_until_all_present(files, PAUSE_TIME, invert)
%Wait until all files are present (or all absent), sleep very
%PAUSE_TIME seconds (Defaults to 5)

if ~exist('PAUSE_TIME','var')
  PAUSE_TIME = 5;
end

TARGET = 0;

if ~exist('invert','var')
  invert = 0;
elseif invert==1
  TARGET = length(files);
end

while 1
  missingfile = cellfun(@(x)~fileexists(x),files);
  if sum(missingfile) == TARGET
    break;
  else
    fprintf(1,['%03d File(s) missing [should be %d', ...
               'waiting %d sec until re-try\n'], ...
               sum(missingfile),TARGET, PAUSE_TIME);
    pause(PAUSE_TIME);
  end
end

