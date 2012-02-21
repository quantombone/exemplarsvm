function rec = PASreadrecord(path)

if length(path)<4
    error('unable to determine format: %s',path);
end

try
  if strcmp(path(end-3:end),'.txt')
    rec=PASreadrectxt(path);
  else
    rec=VOCreadrecxml(path);
  end
  fprintf(1,'.');
catch
  %If file did not load (it is not there, or invalid format), then
  %simply return an empty array (but do not crash!)
  rec = [];
  fprintf(1,',');
end
