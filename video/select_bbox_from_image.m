function bbox = select_bbox_from_image(I, message)
%Given an open figure, display an image then ask for a rectangular
%selection and return the selection in bbox format, while not
%allowing the selection to be too small (area of 50 pixels is minimum)
% Tomasz Malisiewicz (tomasz@csail.mit.edu)

if ~exist('message','var')
  message = sprintf('Select Rectangular Region');
end

while 1
  clf
  imagesc(convert_to_I(I))
  axis image
  axis off
  htitle = title(message);
  set(htitle,'FontSize',18,'FontWeight','bold');
  fprintf(1,['Click a corner, hold until diagonally opposite corner,' ...
             ' and release\n']);
  h = imrect;
  bbox = getPosition(h);

  %This prevents bad clicks, without a hold
  if (bbox(3)*bbox(4) < 50)
    fprintf(1,'Region too small, try again\n');
  else
    break;
  end
end

bbox(3) = bbox(3) + bbox(1);
bbox(4) = bbox(4) + bbox(2);
bbox = round(bbox);
