function bg = get_screenshot_bg(NFRAMES, post_func)
%Get a virtual dataset which on access of a frame, captures a
%screenshot from the device and chooses a subwindow  based on the coordinates
%in capture_region.mat, then applies the "post-processing function" post_func
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

try
  load capture_region.mat
catch
  fprintf(1,'Getting init screenshot in 4 sec\n');
  pause(4)
  capture_region = initialize_screenshot;
end

bg = cell(NFRAMES,1);
if exist('post_func','var')
  for i = 1:length(bg)
    bg{i} = get_f_handle(capture_region, post_func);
  end
else
  for i = 1:length(bg)
    bg{i} = get_f_handle(capture_region);
  end
end

function f = get_f_handle(capture_region, post_func)
if ~exist('post_func','var')
  f = @(x)capture_screen(capture_region);
else
  f = @(x)post_func(capture_screen(capture_region));
end
