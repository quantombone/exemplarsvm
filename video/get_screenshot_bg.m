function bg = get_screenshot_bg(NFRAMES,func)
%Get a virtual dataset which on access of a frame, captures a
%screenshot from the device and indexes it based on the coordinates
%in subber.mat

load subber.mat
%fprintf(1,'Getting init screenshot in 4 sec\n');
%pause(4)
%subber=initialize_screenshot;

bg = cell(NFRAMES,1);
if exist('func','var')
  for i = 1:length(bg)
    bg{i} = getit(subber,func);
  end
else
  for i = 1:length(bg)
    bg{i} = getit(subber);
  end
end

function f = getit(subber,func)
if ~exist('func','var')
  f = @(x)ScreenCapture(subber);
else
  f = @(x)func(ScreenCapture(subber));
end