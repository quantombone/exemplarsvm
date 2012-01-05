function capture_region = initialize_screenshot
%Show a full screne screenshot captured from the screen, then let user
%select a selection region (choose top left corner then drag to bottom
%right corner):
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

I = capture_screen;
capture_region = select_bbox_from_image(I,'Select screenshot region');

if nargout == 0
  save capture_region.mat capture_region
end
