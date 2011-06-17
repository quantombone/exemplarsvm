function wait_until_all_present(files, PAUSE_TIME)
%Wait until all files are present, sleep very PAUSE_TIME seconds
%(Defaults to 5)

if ~exist('PAUSE_TIME','var')
  PAUSE_TIME = 5;
end
if ~isempty(files)
  while 1
    missingfile = cellfun(@(x)~fileexists(x),files);
    if sum(missingfile) == 0
      break;
    else
      fprintf(1,'%03d File(s) missing, waiting %d sec until re-try\n', ...
              sum(missingfile),PAUSE_TIME);
      pause(PAUSE_TIME);
    end
  end
end
