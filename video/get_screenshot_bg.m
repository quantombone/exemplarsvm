function bg = get_screenshot_bg(NFRAMES,func)
bg = cell(NFRAMES,1);
subber=initialize_screenshot;

if exist('func','var')
  for i = 1:length(bg)
    bg{i} = @(x)func(ScreenCapture(subber));
  end
else
  for i = 1:length(bg)
    bg{i} = @(x)ScreenCapture(subber);
  end
end