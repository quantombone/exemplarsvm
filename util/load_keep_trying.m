function res = load_keep_trying(final_file,PAUSE_TIME)
%Keep trying to load a file until the load succeeds waiting
%PAUSE_TIME before re-trying

if ~exist('PAUSE_TIME','var')
  PAUSE_TIME = 5.0;
end

while 1
  try
    res = load(final_file);
    break;
  catch
    fprintf(1,'cannot load %s\n ---sleeping for %.3fsec, trying again\n',...
            final_file,PAUSE_TIME);
    pause(PAUSE_TIME);
  end
end
